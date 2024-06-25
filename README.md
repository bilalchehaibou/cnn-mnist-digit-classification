# MNIST Digit Classification


This repository contains implementations of two models for classifying handwritten digits from the MNIST dataset. One model is a simple Neural Network (NN), and the other is a Convolutional Neural Network (CNN). Both models present a simple architecture and are implemented using TensorFlow and Keras.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
  - [Simple Neural Network](#simple-neural-network)
  - [Convolutional Neural Network](#convolutional-neural-network)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Introduction

The MNIST dataset is a well-known dataset in the field of machine learning. It consists of 70,000 images of handwritten digits (0-9). This project demonstrates how to build and train a simple NN and a CNN to classify these digits.

## Dataset

The MNIST dataset is included in the Keras library, so there is no need to download it separately. It contains 60,000 training images and 10,000 testing images, each of size 28x28 pixels in grayscale.

## Models

### Simple Neural Network

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
                                                                 
 dense (Dense)               (None, 100)               78500     
                                                                 
 batch_normalization (Batch  (None, 100)               400       
 Normalization)                                                  
                                                                 
 dropout (Dropout)           (None, 100)               0         
                                                                 
 dense_1 (Dense)             (None, 80)                8080      
                                                                 
 batch_normalization_1 (Bat  (None, 80)                320       
 chNormalization)                                                
                                                                 
 dropout_1 (Dropout)         (None, 80)                0         
                                                                 
 dense_2 (Dense)             (None, 50)                4050      
                                                                 
 dense_3 (Dense)             (None, 10)                510       
                                                                 
=================================================================
Total params: 91860 (358.83 KB)
Trainable params: 91500 (357.42 KB)
Non-trainable params: 360 (1.41 KB)

### Convolutional Neural Network
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496     
                                                                 
 max_pooling2d (MaxPooling2  (None, 8, 8, 64)          0         
 D)                                                              
                                                                 
 flatten_1 (Flatten)         (None, 4096)              0         
                                                                 
 dense_4 (Dense)             (None, 128)               524416    
                                                                 
 dropout_2 (Dropout)         (None, 128)               0         
                                                                 
 dense_5 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 544522 (2.08 MB)
Trainable params: 544522 (2.08 MB)
Non-trainable params: 0 (0.00 Byte)

### Files 
models_creation.ipynb : jupyter notebook where the models are trained and then saved in models folder
main.py : python script using the CNN model to predict the digit from the picture 3_mnist.png

### Installation

pip install -r requirements.txt


### Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mnist-digit-classification.git
```

2. Navigate to the project directory:

```bash
cd mnist-digit-classification
```

3. Run the training script for the CNN:

```bash
python main.py
```

## Results

Accuracy on unseen data for CNN : 0.9922; for simple NN: 0.9696

