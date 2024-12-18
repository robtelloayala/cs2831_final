import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import tensorflow as tf

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # For images (continuous values): we can use bilinear interpolation by default.
        # For masks (categorical values): use nearest interpolation and fill_mode='nearest'.

        self.augment_inputs = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode="horizontal", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.1, seed=seed),
            tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=seed),
            tf.keras.layers.RandomTranslation(0.1, 0.1, seed=seed)
        ])
        
        self.augment_labels = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode="horizontal", seed=seed),
            # Use nearest interpolation and nearest fill mode to avoid label distortions
            tf.keras.layers.RandomRotation(factor=0.1, interpolation='nearest', fill_mode='nearest', seed=seed),
            tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), 
                                       interpolation='nearest', fill_mode='nearest', seed=seed),
            tf.keras.layers.RandomTranslation(0.1, 0.1, interpolation='nearest', fill_mode='nearest', seed=seed)
        ])
        
    def call(self, inputs, labels):
        # Apply the same random transformations to images and masks
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels

class Augment_Less(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels
  
def plot_and_save_history(history, csv_filename='training_history.csv', fig_filename='training_plot.png', title=None):
    # Convert the history dictionary to a Pandas DataFrame
    df = pd.DataFrame(history.history)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)
    print(f"Training history saved to {csv_filename}")

    # Create a figure with 2 subplots: one for accuracy, one for loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Accuracy
    if 'accuracy' in history.history:
        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # Plot Loss
    ax[1].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title(f'Training and Validation Loss for {title}')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(fig_filename)
    print(f"Training plot saved to {fig_filename}")
    plt.show()

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # For CPU/GPU determinism (depending on TF version & GPU model):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(
    datapoint['segmentation_mask'],
    (128, 128),
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

# for images, masks in train_batches.take(2):
#   sample_image, sample_mask = images[0], masks[0]
#   display([sample_image, sample_mask])

def create_mask(pred_mask):
    # Convert prediction probabilities to class labels
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

#%%
# # Loop through a few batches from the training dataset
# for images, masks in train_batches.take(5):
#     # Select the first image and mask from the batch
#     sample_image, sample_mask = images[0], masks[0]
    
#     # Expand dimensions to match the model's input shape (add batch dimension)
#     sample_image_expanded = tf.expand_dims(sample_image, axis=0)
    
#     # Get the model's prediction for the sample image
#     pred_mask = model.predict(sample_image_expanded)
    
#     # Process the prediction to get the predicted mask
#     pred_mask_processed = create_mask(pred_mask)
    
#     # Display the input image, true mask, and predicted mask
#     display([sample_image, sample_mask, pred_mask_processed])
