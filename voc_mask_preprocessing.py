#%%
import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from training_utils import set_global_seed
# taken from https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/remove_gt_colormap.py
set_global_seed(seed=42)

# Define flags (or equivalent variables) for preprocessing
ORIGINAL_GT_FOLDER = 'VOCdevkit/VOC2012/SegmentationClass'
SEGMENTATION_FORMAT = 'png'
OUTPUT_DIR = 'VOCdevkit/VOC2012/SegmentationClassRaw'

# Create the output directory if not exists
if not tf.io.gfile.isdir(OUTPUT_DIR):
    tf.io.gfile.makedirs(OUTPUT_DIR)

def _remove_colormap(filename):
    """Removes the color map from the annotation."""
    return np.array(Image.open(filename))  # This will give a class-indexed mask if already palette-indexed

def _save_annotation(annotation, filename):
    """Saves the annotation as a png file."""
    pil_image = Image.fromarray(annotation.astype(np.uint8))
    with tf.io.gfile.GFile(filename, mode='w') as f:
        pil_image.save(f, 'PNG')

# Preprocess all segmentation masks in the original folder
annotations = glob.glob(os.path.join(ORIGINAL_GT_FOLDER, '*.' + SEGMENTATION_FORMAT))
for annotation in annotations:
    raw_annotation = _remove_colormap(annotation)
    filename = os.path.basename(annotation)[:-4]
    _save_annotation(raw_annotation, os.path.join(OUTPUT_DIR, filename + '.' + SEGMENTATION_FORMAT))
