#%%
from unet_models import unet_model_batch
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
from training_utils import Augment_Less, plot_and_save_history, set_global_seed
from collections import Counter

VOC_ROOT = "VOCdevkit/VOC2012"
IMAGE_DIR = os.path.join(VOC_ROOT, "JPEGImages")
LABEL_DIR = os.path.join(VOC_ROOT, "SegmentationClassRaw")
TRAIN_LIST = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", "train.txt")
VAL_LIST   = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", "val.txt")

    
set_global_seed(42)

def load_image_list(txt_path):
    with open(txt_path, 'r') as f:
        image_ids = f.read().strip().split()
    return image_ids

train_ids = load_image_list(TRAIN_LIST)
val_ids   = load_image_list(VAL_LIST)

def load_image(image_id):
    img_path = tf.strings.join([IMAGE_DIR, "/", image_id, ".jpg"])
    mask_path = tf.strings.join([LABEL_DIR, "/", image_id, ".png"])

    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.int32)


    image = tf.image.resize(image, (128, 128))
    mask  = tf.image.resize(mask, (128, 128), method='nearest')

    return image, mask

def create_dataset(image_ids, batch_size, shuffle=False, buffer_size=1000):
    dataset = tf.data.Dataset.from_tensor_slices(image_ids)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Function to calculate class frequencies
def calculate_class_frequencies(image_ids, label_dir, num_classes=21):
    class_counts = Counter()
    for image_id in image_ids:
        mask_path = os.path.join(label_dir, f"{image_id}.png")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.cast(mask, tf.int32)
        mask = tf.reshape(mask, [-1])
        class_counts.update(mask.numpy().tolist())
    return class_counts

# train_class_counts = calculate_class_frequencies(train_ids, LABEL_DIR, num_classes=21)
# print("Train Class Counts:", train_class_counts)

# # Compute class weights using Inverse Frequency
# class_weights = {}
# total_pixels = sum(train_class_counts.values())
# for cls in range(21):
#     class_weights[cls] = total_pixels / (21 * train_class_counts[cls])

# # Optionally normalize weights
# mean_weight = np.mean(list(class_weights.values()))
# for cls in class_weights:
#     class_weights[cls] /= mean_weight

# print("Class Weights:", class_weights)
# #%%
class_weights = {i: 1.0 for i in range(1,21)}
class_weights[0] = 0.5
# Define the weighting function
def add_sample_weights(image, mask, class_weights, ignore_class=255):
    # Ensure mask is int32
    mask = tf.cast(mask, tf.int32)
    mask_squeezed = tf.squeeze(mask, axis=-1) 
 
    mask_for_gather = tf.where(
        tf.equal(mask_squeezed, ignore_class),
        tf.zeros_like(mask_squeezed),  # placeholder index = 0
        mask_squeezed
    )

    # Build a float32 tensor of class weights
    weights = tf.constant(
        [class_weights[c] for c in range(len(class_weights))], dtype=tf.float32
    )

    # Gather sample weights (no error now)
    sample_weights = tf.gather(weights, mask_for_gather)

    # Finally, zero out the ignored pixels
    ignore_mask = tf.equal(mask_squeezed, ignore_class)
    sample_weights = tf.where(ignore_mask, 0.0, sample_weights)

    return (image, mask, sample_weights)

def add_sample_weights_wrapper(image, mask):
    return add_sample_weights(image, mask, class_weights=class_weights, ignore_class=255)


BATCH_SIZE = 16
EPOCHS = 40
OUTPUT_CLASSES = 21

augment_layer = Augment_Less(seed=42)

train_dataset = create_dataset(train_ids, batch_size=BATCH_SIZE, shuffle=True).repeat()
train_dataset = train_dataset.map(lambda x, y: augment_layer(x, y), num_parallel_calls=tf.data.AUTOTUNE)

val_dataset = create_dataset(val_ids, batch_size=BATCH_SIZE, shuffle=False)

TRAIN_LENGTH = len(train_ids)
VAL_LENGTH   = len(val_ids)

STEPS_PER_EPOCH   = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS  = VAL_LENGTH // BATCH_SIZE

unet_model_pretrained = unet_model_batch(
    input_size=(128, 128, 3), 
    output_channels=OUTPUT_CLASSES
)

def masked_weighted_sparse_categorical_crossentropy(y_true, y_pred, class_weights, ignore_class=255):

    if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)  

    # Build the per-class weight vector
    num_classes = len(class_weights)
    weights_vec = tf.constant([class_weights[c] for c in range(num_classes)], dtype=tf.float32)

    # Convert cross-entropy to elementwise (reduction='none')
    scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
    ce_per_pixel = scce(y_true, y_pred) 

    valid_mask = tf.not_equal(y_true, ignore_class)  

    y_true_clamped = tf.where(valid_mask, y_true, 0) 
    pixel_class_weights = tf.gather(weights_vec, tf.cast(y_true_clamped, tf.int32)) 

    # Weighted CE = ce_per_pixel * pixel_class_weights
    ce_per_pixel = ce_per_pixel * pixel_class_weights

    # Zero out the invalid pixels
    ce_per_pixel = tf.where(valid_mask, ce_per_pixel, 0.0)

    # Average only over valid pixels
    valid_count = tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + 1e-7
    ce = tf.reduce_sum(ce_per_pixel) / valid_count
    return ce

def masked_weighted_dice_loss_multiclass(y_true, y_pred, class_weights, ignore_class=255, smooth=1e-6):
    if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)  

    valid_mask = tf.not_equal(y_true, ignore_class)
    valid_mask_4d = tf.cast(tf.expand_dims(valid_mask, axis=-1), tf.float32)

    # Force ignored pixels to some valid label just for one-hot
    y_true_clamped = tf.where(valid_mask, y_true, 0)
    y_true_clamped = tf.cast(y_true_clamped, tf.int32)

    num_classes = 21  
    y_true_onehot = tf.one_hot(y_true_clamped, depth=num_classes)

    y_pred_prob = tf.nn.softmax(y_pred, axis=-1)

    weights_list = [class_weights[c] for c in range(num_classes)] 
    weights_vec = tf.constant(weights_list, dtype=tf.float32)  
    weights_broadcast = tf.reshape(weights_vec, [1, 1, 1, num_classes])

    intersection = tf.reduce_sum(valid_mask_4d * y_true_onehot * y_pred_prob * weights_broadcast, axis=[0,1,2])
    denom = (tf.reduce_sum(valid_mask_4d * y_true_onehot * weights_broadcast, axis=[0,1,2])
             + tf.reduce_sum(valid_mask_4d * y_pred_prob * weights_broadcast, axis=[0,1,2]))

    dice_per_class = (2.0 * intersection + smooth) / (denom + smooth)
    dice_mean = tf.reduce_mean(dice_per_class)
    return 1.0 - dice_mean

def masked_weighted_combined_loss(y_true, y_pred, class_weights, ignore_class=255):
    ce = masked_weighted_sparse_categorical_crossentropy(
        y_true, y_pred, class_weights, ignore_class
    )
    dice = masked_weighted_dice_loss_multiclass(
        y_true, y_pred, class_weights, ignore_class
    )
    return ce + dice

def masked_weighted_combined_loss_wrapper(y_true, y_pred):
    return masked_weighted_combined_loss(
        y_true, y_pred, 
        class_weights=class_weights, 
        ignore_class=255
    )

unet_model_pretrained.compile(
     optimizer='adam',
     loss=masked_weighted_combined_loss_wrapper,
     metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(
        filepath='unet_models_voc_less_augmentation/model_unet_batch_no_weights.keras', 
        monitor='val_loss',
        save_best_only=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    )
]

history = unet_model_pretrained.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_dataset,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks
)

plot_and_save_history(
    history, 
    csv_filename='unet_models_voc_less_augmentation/training_history_batch_unet_no_weights_03.csv', 
    fig_filename='unet_models_voc_less_augmentation/training_history_batch_unet_no_weights_03.png', 
    title='UNet with Batch Normalization'
)
