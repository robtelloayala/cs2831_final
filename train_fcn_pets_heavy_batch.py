#%%
from FCN_models import fcn_batch
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from training_utils import Augment, plot_and_save_history, set_global_seed, load_image

set_global_seed(seed=42)

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
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)

model = fcn_batch(input_size=(128, 128, 3), output_channels=OUTPUT_CLASSES)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE

history = model.fit(
    train_batches,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=test_batches,
    validation_steps=VALIDATION_STEPS,
    callbacks=[ModelCheckpoint(filepath='fcn_model_pet_heavy_augmentation/model_fcn_batch.keras', 
                                         monitor='val_loss',
                                         save_best_only=True, 
                                         val_metric='val_loss'),
                        ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.5,
                                            patience=3)]
)

plot_and_save_history(history, csv_filename='fcn_model_pet_heavy_augmentation/training_history_fcn_batch.csv', fig_filename='fcn_model_pet_heavy_augmentation/training_history_fcn_batch.png', title='FCN with BatchNorm')

# %%