#%%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from training_utils import load_image, Augment_Less
import pandas as pd

tf.keras.utils.set_random_seed(42)

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
EPOCHS = 40
OUTPUT_CLASSES = 3

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment_Less())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)


def compute_miou_in_batches(model, dataset, output_classes):
    """
    Compute the Mean IoU on a given dataset in batches 
    to avoid memory overload.
    """
    miou_metric = tf.keras.metrics.MeanIoU(num_classes=output_classes)
    for imgs, masks in dataset:
        # Make predictions
        preds = model.predict(imgs)  # shape: [batch_size, height, width, output_classes]
        pred_class = np.argmax(preds, axis=-1)  # shape: [batch_size, height, width]
        
        # Convert ground truth mask to correct shape if needed
        true_class = np.squeeze(masks.numpy(), axis=-1)  # shape: [batch_size, height, width]
        
        # Update MeanIoU state
        miou_metric.update_state(true_class, pred_class)
        
    return miou_metric.result().numpy()


models_dict = {
    "Batch UNet": "unet_models_pets_little_augmentation/training_history_batch_unet.csv",
    "UNet": "unet_models_pets_little_augmentation/training_history_unet.csv",
    "LayerNorm UNet": "unet_models_pets_little_augmentation/training_history_layernorm_unet.csv",
    "UNet with Pretrained encoder": "unet_models_pets_little_augmentation/training_history_pretrained_unet.csv"
}


def load_model_function(model_name):
    """Loads and returns the corresponding model from disk."""
    if model_name == "Batch UNet":
        return tf.keras.models.load_model("unet_models_pets_little_augmentation/model_batch_unet.keras")
    elif model_name == "UNet":
        return tf.keras.models.load_model("unet_models_pets_little_augmentation/model_unet.keras")
    elif model_name == "LayerNorm UNet":
        return tf.keras.models.load_model("unet_models_pets_little_augmentation/model_layernorm_unet.keras")
    elif model_name == "UNet with Pretrained encoder":
        return tf.keras.models.load_model("unet_models_pets_little_augmentation/model_pretrained_unet.keras")

    return None

miou_results = {}
for model_name in models_dict.keys():
    model = load_model_function(model_name)
    if model is None:
        print(f"Model not found for '{model_name}' - skipping.")
        continue
    miou_score = compute_miou_in_batches(model, test_batches, OUTPUT_CLASSES)
    miou_results[model_name] = miou_score


sns.set_style("whitegrid")

model_names = list(miou_results.keys())
miou_scores = list(miou_results.values())

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(model_names, miou_scores, color=colors[:len(model_names)], alpha=0.9)

ax.set_title("Mean IoU Comparison", fontsize=16, fontweight='bold')
ax.set_xlabel("Models", fontsize=14)
ax.set_ylabel("Mean IoU", fontsize=14)
ax.set_ylim(0, 1.0) 
ax.tick_params(axis='both', which='major', labelsize=12)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("unet_models_pets_little_augmentation/miou_comparison.png", dpi=300)
plt.show()

miou_df = pd.DataFrame(list(miou_results.items()), columns=['Model', 'Mean_IoU'])


csv_file_path = "unet_models_pets_little_augmentation/miou_results.csv"

miou_df.to_csv(csv_file_path, index=False)
print(f"Mean IoU results saved to {csv_file_path}")
#%%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from training_utils import load_image, Augment_Less

tf.keras.utils.set_random_seed(42)


dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
EPOCHS = 40
OUTPUT_CLASSES = 3

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment_Less())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)
# ------------------------------
# 2) DEFINE A FUNCTION TO COMPUTE MIOU IN BATCHES
# ------------------------------
def compute_miou_in_batches(model, dataset, output_classes):
    """
    Compute the Mean IoU on a given dataset in batches 
    to avoid memory overload.
    """
    miou_metric = tf.keras.metrics.MeanIoU(num_classes=output_classes)
    for imgs, masks in dataset:
        # Make predictions
        preds = model.predict(imgs)  # shape: [batch_size, height, width, output_classes]
        pred_class = np.argmax(preds, axis=-1)  # shape: [batch_size, height, width]
        
        # Convert ground truth mask to correct shape if needed
        true_class = np.squeeze(masks.numpy(), axis=-1)  # shape: [batch_size, height, width]
        
        # Update MeanIoU state
        miou_metric.update_state(true_class, pred_class)
        
    return miou_metric.result().numpy()

# ------------------------------
# 3) DEFINE DICTIONARY OF MODEL NAMES AND CSV HISTORY PATHS
# ------------------------------
models_dict = {
    "Batch UNet": "unet_models_pets_heavy_augmentation/training_history_batch_unet.csv",
    "UNet": "unet_models_pets_heavy_augmentation/training_history_unet.csv",
    "LayerNorm UNet": "unet_models_pets_heavy_augmentation/training_history_layernorm_unet.csv",
    "UNet with Pretrained encoder": "unet_models_pets_heavy_augmentation/training_history_pretrained_unet.csv"
}

# ------------------------------
# 4) DEFINE A FUNCTION TO LOAD THE TRAINED MODELS
# ------------------------------
def load_model_function(model_name):
    """Loads and returns the corresponding model from disk."""
    if model_name == "Batch UNet":
        return tf.keras.models.load_model("unet_models_pets_heavy_augmentation/model_batch_unet.keras")
    elif model_name == "UNet":
        return tf.keras.models.load_model("unet_models_pets_heavy_augmentation/model_unet.keras")
    elif model_name == "LayerNorm UNet":
        return tf.keras.models.load_model("unet_models_pets_heavy_augmentation/model_layernorm_unet.keras")
    elif model_name == "UNet with Pretrained encoder":
        return tf.keras.models.load_model("unet_models_pets_heavy_augmentation/model_pretrained_unet.keras")
    # Add any additional model logic if needed
    return None

# ------------------------------
# 5) COMPUTE MIOU FOR EACH MODEL
# ------------------------------
miou_results = {}
for model_name in models_dict.keys():
    model = load_model_function(model_name)
    if model is None:
        print(f"Model not found for '{model_name}' - skipping.")
        continue
    miou_score = compute_miou_in_batches(model, test_batches, OUTPUT_CLASSES)
    miou_results[model_name] = miou_score

# ------------------------------
# 6) PLOT THE RESULTS
# ------------------------------
sns.set_style("whitegrid")

model_names = list(miou_results.keys())
miou_scores = list(miou_results.values())

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(model_names, miou_scores, color=colors[:len(model_names)], alpha=0.9)

ax.set_title("Mean IoU Comparison", fontsize=16, fontweight='bold')
ax.set_xlabel("Models", fontsize=14)
ax.set_ylabel("Mean IoU", fontsize=14)
ax.set_ylim(0, 1.0)  # IoU typically ranges from 0 to 1
ax.tick_params(axis='both', which='major', labelsize=12)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("unet_models_pets_heavy_augmentation/miou_comparison.png", dpi=300)
plt.show()

# Convert the miou_results dictionary to a pandas DataFrame
miou_df = pd.DataFrame(list(miou_results.items()), columns=['Model', 'Mean_IoU'])

# Define the CSV file path
csv_file_path = "unet_models_pets_heavy_augmentation/miou_results.csv"

# Save the DataFrame to a CSV file
miou_df.to_csv(csv_file_path, index=False)
print(f"Mean IoU results saved to {csv_file_path}")
# %%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from training_utils import load_image, Augment_Less

tf.keras.utils.set_random_seed(42)


dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
EPOCHS = 40
OUTPUT_CLASSES = 3

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment_Less())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)

def compute_miou_in_batches(model, dataset, output_classes):
    """
    Compute the Mean IoU on a given dataset in batches 
    to avoid memory overload.
    """
    miou_metric = tf.keras.metrics.MeanIoU(num_classes=output_classes)
    for imgs, masks in dataset:
        # Make predictions
        preds = model.predict(imgs)  # shape: [batch_size, height, width, output_classes]
        pred_class = np.argmax(preds, axis=-1)  # shape: [batch_size, height, width]
        
        # Convert ground truth mask to correct shape if needed
        true_class = np.squeeze(masks.numpy(), axis=-1)  # shape: [batch_size, height, width]
        
        # Update MeanIoU state
        miou_metric.update_state(true_class, pred_class)
        
    return miou_metric.result().numpy()

models_dict = {
    "Batch FCN": "fcn_model_pet_less_batch_augmentation/training_history_fcn_batch.csv",
    "FCN": "fcn_model_pet_less_batch_augmentation/training_history_fcn_batch.csv",
    "LayerNorm FCN": "fcn_model_pet_less_batch_augmentation/training_history_fcn_norm.csv",
}


def load_model_function(model_name):
    """Loads and returns the corresponding model from disk."""
    if model_name == "Batch FCN":
        return tf.keras.models.load_model("fcn_model_pet_less_augmentation/model_fcn_batch.keras")
    elif model_name == "FCN":
        return tf.keras.models.load_model("fcn_model_pet_less_augmentation/model_fcn.keras")
    elif model_name == "LayerNorm FCN":
        return tf.keras.models.load_model("fcn_model_pet_less_augmentation/model_fcn_norm.keras")
    return None

miou_results = {}
for model_name in models_dict.keys():
    model = load_model_function(model_name)
    if model is None:
        print(f"Model not found for '{model_name}' - skipping.")
        continue
    miou_score = compute_miou_in_batches(model, test_batches, OUTPUT_CLASSES)
    miou_results[model_name] = miou_score

sns.set_style("whitegrid")

model_names = list(miou_results.keys())
miou_scores = list(miou_results.values())

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(model_names, miou_scores, color=colors[:len(model_names)], alpha=0.9)

ax.set_title("Mean IoU Comparison", fontsize=16, fontweight='bold')
ax.set_xlabel("Models", fontsize=14)
ax.set_ylabel("Mean IoU", fontsize=14)
ax.set_ylim(0, 1.0) 
ax.tick_params(axis='both', which='major', labelsize=12)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("fcn_model_pet_less_augmentation/miou_comparison.png", dpi=300)
plt.show()

miou_df = pd.DataFrame(list(miou_results.items()), columns=['Model', 'Mean_IoU'])

csv_file_path = "fcn_model_pet_less_augmentation/miou_results.csv"


miou_df.to_csv(csv_file_path, index=False)
print(f"Mean IoU results saved to {csv_file_path}")
# %%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from training_utils import load_image, Augment_Less
import pandas as pd

tf.keras.utils.set_random_seed(42)

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
EPOCHS = 40
OUTPUT_CLASSES = 3

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment_Less())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)

def compute_miou_in_batches(model, dataset, output_classes):
    """
    Compute the Mean IoU on a given dataset in batches 
    to avoid memory overload.
    """
    miou_metric = tf.keras.metrics.MeanIoU(num_classes=output_classes)
    for imgs, masks in dataset:
        # Make predictions
        preds = model.predict(imgs)  # shape: [batch_size, height, width, output_classes]
        pred_class = np.argmax(preds, axis=-1)  # shape: [batch_size, height, width]
        
        # Convert ground truth mask to correct shape if needed
        true_class = np.squeeze(masks.numpy(), axis=-1)  # shape: [batch_size, height, width]
        
        # Update MeanIoU state
        miou_metric.update_state(true_class, pred_class)
        
    return miou_metric.result().numpy()


models_dict = {
    "Batch FCN": "fcn_model_pet_heavy_augmentation/training_history_fcn_batch.csv",
    "FCN": "fcn_model_pet_heavy_augmentation/training_history_fcn_batch.csv",
    "LayerNorm FCN": "fcn_model_pet_heavy_augmentation/training_history_fcn_norm.csv",
}


def load_model_function(model_name):
    """Loads and returns the corresponding model from disk."""
    if model_name == "Batch FCN":
        return tf.keras.models.load_model("fcn_model_pet_heavy_augmentation/model_fcn_batch.keras")
    elif model_name == "FCN":
        return tf.keras.models.load_model("fcn_model_pet_heavy_augmentation/model_fcn.keras")
    elif model_name == "LayerNorm FCN":
        return tf.keras.models.load_model("fcn_model_pet_heavy_augmentation/model_fcn_norm.keras")
    return None


miou_results = {}
for model_name in models_dict.keys():
    model = load_model_function(model_name)
    if model is None:
        print(f"Model not found for '{model_name}' - skipping.")
        continue
    miou_score = compute_miou_in_batches(model, test_batches, OUTPUT_CLASSES)
    miou_results[model_name] = miou_score

sns.set_style("whitegrid")

model_names = list(miou_results.keys())
miou_scores = list(miou_results.values())

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(model_names, miou_scores, color=colors[:len(model_names)], alpha=0.9)

ax.set_title("Mean IoU Comparison", fontsize=16, fontweight='bold')
ax.set_xlabel("Models", fontsize=14)
ax.set_ylabel("Mean IoU", fontsize=14)
ax.set_ylim(0, 1.0)  
ax.tick_params(axis='both', which='major', labelsize=12)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("fcn_model_pet_heavy_augmentation/miou_comparison.png", dpi=300)
plt.show()

miou_df = pd.DataFrame(list(miou_results.items()), columns=['Model', 'Mean_IoU'])

csv_file_path = "fcn_model_pet_heavy_augmentation/miou_results.csv"

miou_df.to_csv(csv_file_path, index=False)
print(f"Mean IoU results saved to {csv_file_path}")
# %%
# %%
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from training_utils import load_image, Augment_Less
import pandas as pd

tf.keras.utils.set_random_seed(42)

from training_utils import set_global_seed

VOC_ROOT = "/Users/roberto/Downloads/VOCdevkit/VOC2012"
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


BATCH_SIZE = 16
EPOCHS = 40
OUTPUT_CLASSES = 21

augment_layer = Augment_Less(seed=42)

val_dataset = create_dataset(val_ids, batch_size=BATCH_SIZE, shuffle=False)

VAL_LENGTH   = len(val_ids)

VALIDATION_STEPS  = VAL_LENGTH // BATCH_SIZE


def compute_miou_in_batches(model, dataset, output_classes):
    miou_metric = tf.keras.metrics.MeanIoU(num_classes=output_classes)
    for imgs, masks in dataset:
        # Make predictions
        preds = model.predict(imgs)  # shape: [batch_size, height, width, output_classes]
        pred_class = np.argmax(preds, axis=-1)  # shape: [batch_size, height, width]

        # Convert ground truth mask to correct shape if needed
        true_class = np.squeeze(masks.numpy(), axis=-1)  # shape: [batch_size, height, width]

        # Create a mask for valid pixels (not ignore_class=255)
        valid_mask = (true_class != 255)

        # Filter out ignore pixels
        true_class_valid = true_class[valid_mask]
        pred_class_valid = pred_class[valid_mask]

        # Update MeanIoU state only with valid pixels
        miou_metric.update_state(true_class_valid, pred_class_valid)

    return miou_metric.result().numpy()

models_dict = {
    "Batch FCN Class Weights": "fcn_model_voc_less_augmentation/training_history_fcn_batch.csv",
    "FCN Class Weights": "fcn_model_voc_less_augmentation/training_history_fcn_batch.csv",
    "LayerNorm FCN Class Weights": "fcn_model_voc_less_augmentation/training_history_fcn_norm.csv",
    "Batch FCN Foreground/Background Weights": "fcn_model_voc_less_augmentation/training_history_fcn_batch.csv",
    "FCN Foreground/Background Weights": "fcn_model_voc_less_augmentation/training_history_fcn_batch.csv",
    "LayerNorm FCN Foreground/Background Weights": "fcn_model_voc_less_augmentation/training_history_fcn_norm.csv"

}

def load_model_function(model_name):
    """Loads and returns the corresponding model from disk."""
    if model_name == "Batch FCN Class Weights":
        return tf.keras.models.load_model("fcn_model_voc_less_augmentation/model_batch_fcn.keras")
    elif model_name == "FCN Class Weights":
        return tf.keras.models.load_model("fcn_model_voc_less_augmentation/model_fcn.keras")
    elif model_name == "LayerNorm FCN Class Weights":
        return tf.keras.models.load_model("fcn_model_voc_less_augmentation/model_norm_fcn.keras")
    elif model_name == "Batch FCN Foreground/Background Weights":
        return tf.keras.models.load_model("fcn_model_voc_less_augmentation/model_batch_fcn_no_weights.keras")
    elif model_name == "FCN Foreground/Background Weights":
        return tf.keras.models.load_model("fcn_model_voc_less_augmentation/model_fcn_no_weights.keras")
    elif model_name == "LayerNorm FCN Foreground/Background Weights":
        return tf.keras.models.load_model("fcn_model_voc_less_augmentation/model_norm_fcn_no_weights.keras")
    return None

def masked_weighted_sparse_categorical_crossentropy(y_true, y_pred, class_weights, ignore_class=255):
    """
    - Ignores pixels labeled `ignore_class` by zeroing out their loss.
    - Also applies class_weights to handle imbalance.
    
    y_true: (batch, h, w) with values in [0..C-1] or 255.
    y_pred: (batch, h, w, C) logits or probabilities.
    class_weights: dict or list of length C (one weight per class).
    """
    if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)  # now shape is (batch, H, W)

    # Build the per-class weight vector
    num_classes = len(class_weights)
    weights_vec = tf.constant([class_weights[c] for c in range(num_classes)], dtype=tf.float32)

    # Convert cross-entropy to elementwise (reduction='none')
    scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
    ce_per_pixel = scce(y_true, y_pred)  # shape: (batch, h, w)

    # Build valid_mask where label != ignore_class
    valid_mask = tf.not_equal(y_true, ignore_class)  # shape: (batch, h, w)

    # Also gather the class weight for each pixel
    y_true_clamped = tf.where(valid_mask, y_true, 0)  # replace 255 with 0 so gather won't break
    pixel_class_weights = tf.gather(weights_vec, tf.cast(y_true_clamped, tf.int32))  # shape: (batch, h, w)

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
        y_true = tf.squeeze(y_true, axis=-1)  # now shape is (batch, H, W)

    valid_mask = tf.not_equal(y_true, ignore_class)
    valid_mask_4d = tf.cast(tf.expand_dims(valid_mask, axis=-1), tf.float32)

    # Force ignored pixels to some valid label just for one-hot
    y_true_clamped = tf.where(valid_mask, y_true, 0)
    y_true_clamped = tf.cast(y_true_clamped, tf.int32)
    # We already know it's 21
    num_classes = 21   # or just 21 if that's your known constant
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

class_weights = {i: 1.0 for i in range(1,21)}
class_weights[0] = 0.5

@tf.keras.saving.register_keras_serializable()
def masked_weighted_combined_loss_wrapper(y_true, y_pred):
    return masked_weighted_combined_loss(
        y_true, y_pred, 
        class_weights=class_weights, 
        ignore_class=255
    )

miou_results = {}
for model_name in models_dict.keys():
    model = load_model_function(model_name)
    if model is None:
        print(f"Model not found for '{model_name}' - skipping.")
        continue
    miou_score = compute_miou_in_batches(model, val_dataset, OUTPUT_CLASSES)
    miou_results[model_name] = miou_score

sns.set_style("whitegrid")

model_names = list(miou_results.keys())
miou_scores = list(miou_results.values())

colors = colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(model_names, miou_scores, color=colors[:len(model_names)], alpha=0.9)

ax.set_title("Mean IoU Comparison", fontsize=16, fontweight='bold')
ax.set_xlabel("Models", fontsize=14)
ax.set_ylabel("Mean IoU", fontsize=14)
ax.set_ylim(0, 1.0)  # IoU typically ranges from 0 to 1
ax.tick_params(axis='both', which='major', labelsize=12)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("fcn_model_voc_less_augmentation/miou_comparison.png", dpi=300)
plt.show()

miou_df = pd.DataFrame(list(miou_results.items()), columns=['Model', 'Mean_IoU'])

csv_file_path = "fcn_model_voc_less_augmentation/miou_results.csv"

miou_df.to_csv(csv_file_path, index=False)
print(f"Mean IoU results saved to {csv_file_path}")
# %%
# %%
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from training_utils import load_image, Augment_Less
import pandas as pd

tf.keras.utils.set_random_seed(42)

from training_utils import set_global_seed

VOC_ROOT = "/Users/roberto/Downloads/VOCdevkit/VOC2012"
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


BATCH_SIZE = 16
EPOCHS = 40
OUTPUT_CLASSES = 21

augment_layer = Augment_Less(seed=42)

val_dataset = create_dataset(val_ids, batch_size=BATCH_SIZE, shuffle=False)

VAL_LENGTH   = len(val_ids)

VALIDATION_STEPS  = VAL_LENGTH // BATCH_SIZE

def compute_miou_in_batches(model, dataset, output_classes):
    miou_metric = tf.keras.metrics.MeanIoU(num_classes=output_classes)
    for imgs, masks in dataset:
        # Make predictions
        preds = model.predict(imgs)  # shape: [batch_size, height, width, output_classes]
        pred_class = np.argmax(preds, axis=-1)  # shape: [batch_size, height, width]

        # Convert ground truth mask to correct shape if needed
        true_class = np.squeeze(masks.numpy(), axis=-1)  # shape: [batch_size, height, width]

        # Create a mask for valid pixels (not ignore_class=255)
        valid_mask = (true_class != 255)

        # Filter out ignore pixels
        true_class_valid = true_class[valid_mask]
        pred_class_valid = pred_class[valid_mask]

        # Update MeanIoU state only with valid pixels
        miou_metric.update_state(true_class_valid, pred_class_valid)

    return miou_metric.result().numpy()

models_dict = {
    "Batch U-Net Class Weights": "fcn_model_voc_less_augmentation/training_history_fcn_batch.csv",
    "U-Net Class Weights": "fcn_model_voc_less_augmentation/training_history_fcn_batch.csv",
    "LayerNorm U-Net Class Weights": "fcn_model_voc_less_augmentation/training_history_fcn_norm.csv",
    "Pretrained U-Net Class Weights" : "fcn_model_voc_less_augmentation/training_history_fcn_pretrained.csv",
    "Batch U-Net Foreground/Background Weights": "fcn_model_voc_less_augmentation/training_history_fcn_batch.csv",
    "U-Net Foreground/Background Weights": "fcn_model_voc_less_augmentation/training_history_fcn_batch.csv",
    "LayerNorm U-Net Foreground/Background Weights": "fcn_model_voc_less_augmentation/training_history_fcn_norm.csv",
    "Pretrained U-Net Foreground/Background Weights" : "fcn_model_voc_less_augmentation/training_history_fcn_pretrained.csv"
}

def load_model_function(model_name):
    """Loads and returns the corresponding model from disk."""
    if model_name == "Batch U-Net Class Weights":
        return tf.keras.models.load_model("unet_models_voc_less_augmentation/model_unet_batch.keras")
    elif model_name == "U-Net Class Weights":
        return tf.keras.models.load_model("unet_models_voc_less_augmentation/model_unet.keras")
    elif model_name == "LayerNorm U-Net Class Weights":
        return tf.keras.models.load_model("unet_models_voc_less_augmentation/model_unet_norm.keras")
    elif model_name == "Pretrained U-Net Class Weights":
        return tf.keras.models.load_model("unet_models_voc_less_augmentation/model_unet_pretrained.keras")
    elif model_name == "Batch U-Net Foreground/Background Weights":
        return tf.keras.models.load_model("unet_models_voc_less_augmentation/model_unet_batch_no_weights.keras")
    elif model_name == "U-Net Foreground/Background Weights":
        return tf.keras.models.load_model("unet_models_voc_less_augmentation/model_unet_no_weights.keras")
    elif model_name == "LayerNorm U-Net Foreground/Background Weights":
        return tf.keras.models.load_model("unet_models_voc_less_augmentation/model_unet_norm_no_weights.keras")
    elif model_name == "Pretrained U-Net Foreground/Background Weights":
        return tf.keras.models.load_model("unet_models_voc_less_augmentation/model_unet_pretrained_no_weights.keras")
    return None

def masked_weighted_sparse_categorical_crossentropy(y_true, y_pred, class_weights, ignore_class=255):
    if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1) 

    # Build the per-class weight vector
    num_classes = len(class_weights)
    weights_vec = tf.constant([class_weights[c] for c in range(num_classes)], dtype=tf.float32)

    # Convert cross-entropy to elementwise (reduction='none')
    scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
    ce_per_pixel = scce(y_true, y_pred)  

    # Build valid_mask where label != ignore_class
    valid_mask = tf.not_equal(y_true, ignore_class)  

    # Also gather the class weight for each pixel
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

class_weights = {i: 1.0 for i in range(1,21)}
class_weights[0] = 0.5

@tf.keras.saving.register_keras_serializable()
def masked_weighted_combined_loss_wrapper(y_true, y_pred):
    return masked_weighted_combined_loss(
        y_true, y_pred, 
        class_weights=class_weights, 
        ignore_class=255
    )

miou_results = {}
for model_name in models_dict.keys():
    model = load_model_function(model_name)
    if model is None:
        print(f"Model not found for '{model_name}' - skipping.")
        continue
    miou_score = compute_miou_in_batches(model, val_dataset, OUTPUT_CLASSES)
    miou_results[model_name] = miou_score

sns.set_style("whitegrid")

model_names = list(miou_results.keys())
miou_scores = list(miou_results.values())

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#17becf", "#7f7f7f", "#9467bd", "#8c564b"]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(model_names, miou_scores, color=colors[:len(model_names)], alpha=0.9)

ax.set_title("Mean IoU Comparison", fontsize=16, fontweight='bold')
ax.set_xlabel("Models", fontsize=14)
ax.set_ylabel("Mean IoU", fontsize=14)
ax.set_ylim(0, 1.0)  
ax.tick_params(axis='both', which='major', labelsize=12)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("unet_models_voc_less_augmentation/miou_comparison.png", dpi=300)
plt.show()

miou_df = pd.DataFrame(list(miou_results.items()), columns=['Model', 'Mean_IoU'])

csv_file_path = "unet_models_voc_less_augmentation/miou_results.csv"

miou_df.to_csv(csv_file_path, index=False)
print(f"Mean IoU results saved to {csv_file_path}")
# %%
