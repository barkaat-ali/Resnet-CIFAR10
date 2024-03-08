# CIFAR-10 Image Classification with ResNet

This repository contains code for training a ResNet model on the CIFAR-10 dataset using TensorFlow 2. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The objective is to classify images into one of these predefined categories.

## Overview

The CIFAR-10 dataset is loaded and preprocessed, including normalization and data augmentation if enabled. The ResNet architecture is implemented and trained using the training data, with various callbacks such as model checkpointing and learning rate scheduling. The trained model is evaluated on the test dataset to assess its performance.

## Dependencies

- TensorFlow 2
- NumPy
- os
  

4. Monitor training progress and evaluate the model using the generated checkpoints.

## Configuration

- `model_type`: Specifies the type of model to use (e.g., ResNet).
- `batch_size`: Batch size for training.
- `epochs`: Number of training epochs.
- `data_augmentation`: Whether to apply data augmentation during training.
- `num_classes`: Number of classes in the dataset.
- `subtract_pixel_mean`: Whether to subtract the pixel mean from the data.
- `n`: Parameter controlling the depth of the ResNet model.
- `version`: Version of the ResNet model.
- `depth`: Computed depth based on the ResNet model parameters.

## Training

- The CIFAR-10 dataset is loaded and preprocessed, including normalization and optional data augmentation.
- The ResNet model is constructed based on the specified parameters.
- The model is compiled with the categorical cross-entropy loss function and the Adam optimizer with a custom learning rate schedule.
- Training is performed with or without data augmentation, and model checkpoints are saved periodically.
- Learning rate adjustment and model saving are handled using callbacks.

## Evaluation

- The trained model is evaluated on the test dataset to assess its performance.
- Test loss and accuracy metrics are computed and printed to evaluate the model's performance.

## Results

- The trained model achieves a certain 91% accuracy on the test dataset, providing insights into its generalization performance.


