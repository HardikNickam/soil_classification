# README

## Overview
This project is a Flask application that serves as an image classification tool using a pre-trained TensorFlow model. The model classifies different soil types based on uploaded images.

## Features
- Upload images for classification.
- Utilizes a TensorFlow/Keras model for predictions.
- Trains a model on five different soil types.
- Displays prediction results on the same page.

## Requirements
- Python 3.x
- Flask
- TensorFlow (with Keras)
- NumPy
- Pandas

##Training Script Overview

Data Preparation
- Data Generators: The script uses ImageDataGenerator to augment the training data with transformations like rotation, zoom, and shifts.
- Train and Validation Split: The dataset is split into training and validation sets.

Model Architecture
- Convolutional Layers: The model consists of five convolutional layers followed by max pooling layers, flattening, and dense layers.
- Output Layer: The output layer uses softmax activation for multi-class classification (5 soil types).

Training Process
- Optimizer: Adam optimizer is used with a learning rate of 0.001.
- Metrics: The model is evaluated using accuracy, precision, and recall.
- Checkpointing: The best model is saved as best_model.hdf5 based on validation accuracy.
