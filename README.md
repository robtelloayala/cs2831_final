# Repository for Final Project: "Utilizing Neural Networks for Image Segmentation" (CS2831 Advanced Computer Vision)

**Authors:**  
J. Roberto Tello Ayala, Michael Finch, Ryan Nguyen

---

## Overview

This repository contains the code and notebooks used for training and evaluating segmentation models for various datasets, including pets and the VOC dataset. We utilize TensorFlow 2.15.0 for training. Pre-trained models, training history logs, plots, and scripts to reproduce results are included. The VOC dataset is 2gb and could not be includede in the repository. It must be downloaded from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit, dropped in the folder and the working following `cs2831_final` and the following script must be run to preprocess the dataset `voc_mask_preprocessing.py`. Once in the link, the first option must be selected, i.e - Download the training/validation data (2GB tar file). The VOCdevkit should be in the folder, i.e `cs2831_final/VOCdevkit`. The pets dataset can be easily downloaded through the Tensorflow dataset library as shown in the code. 

## Pre-trained Models and Outputs

Pre-trained models (in `.keras` format) and corresponding training histories, CSV logs, and result plots are organized into the following folders:

- `fcn_model_pet_heavy_augmentation`  
  FCN model for pets (Heavy Augmentation)
  
- `fcn_model_pet_less_augmentation`  
  FCN model for pets (Less Augmentation)
  
- `fcn_model_voc_less_augmentation`  
  FCN models for VOC (Both weight schemes)
  
- `unet_models_pets_heavy_augmentation`  
  U-Net models for pets (Heavy Augmentation)
  
- `unet_models_pets_little_augmentation`  
  U-Net models for pets (Less Augmentation)
  
- `unet_models_voc_less_augmentation`  
  U-Net models for VOC (Both weight schemes)

These directories include:
- Trained models
- Training history in `.csv` format
- Training history plots
- Scripts to generate tables and CSV results used in the report

## Notebooks

Three Jupyter notebooks demonstrate how to load models and generate segmentation masks:

- `figures_pets.ipynb`
- `figures_voc.ipynb`
- `cs2831_segformer.ipynb`

## Running the Scripts

To train or run inference, clone the repository and navigate into it via command line:

cd cs2831
python <script_name>.py
For example, to train the U-Net with Batch Normalization on the Pets dataset with Heavy augmentations:

`python train_unet_batch.py`

## Training Scripts

The following scripts train various model architectures (FCN, U-Net) with different normalization schemes (Batch Norm, Layer Norm) and augmentation settings (Heavy, Less), as well as different datasets (Pets, VOC):

**FCN (Pets):**
- `train_fcn_pet_heavy_norm.py`
- `train_fcn_pets_heavy.py`
- `train_fcn_pets_heavy_batch.py`
- `train_fcn_pets_less.py`
- `train_fcn_pets_less_batch.py`
- `train_fcn_pets_less_norm.py`

**U-Net (Pets):**
- `train_uned_pretrained_less.py`
- `train_unet.py`
- `train_unet_batch.py`
- `train_unet_batch_less.py`
- `train_unet_layernorm.py`
- `train_unet_pretrained.py`
- `train_unet_less.py`
- `train_unet_layernorm_less.py`

**U-Net (VOC):**
- `train_voc_unet_weights.py`
- `train_voc_unet_pretrained_weights.py`
- `train_voc_unet_norm_weights.py`
- `train_voc_unet_batch_weights.py`

**FCN (VOC):**
- `train_voc_fcn_weights.py`
- `train_voc_fcn_norm_weights.py`
- `train_voc_fcn_batch_weights.py`

*Note:* For VOC training scripts, to switch between foreground/background weighting scheme and class weighting, uncomment the appropriate lines in the script.

## Other Files

**Model Definitions & Utilities:**
- `FCN_models.py` : FCN model definitions
- `simple_unet.py` : U-Net in PyTorch
- `unet_models.py` : U-Net model definitions

**Training & Preprocessing:**
- `training_utils.py` : Training utilities, augmentation classes, and training history plotting
- `voc_mask_preprocessing.py` : Preprocessing for the VOC dataset

**Evaluation & Comparisons:**
- `miou_comparison.py` : Mean IoU inference comparison
- `comparison_plots.py` : Comparison plots for training history


