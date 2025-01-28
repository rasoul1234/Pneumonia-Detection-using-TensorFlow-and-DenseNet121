# Pneumonia Detection with DenseNet121

This repository contains code for training a convolutional neural network (CNN) model to detect pneumonia in chest X-ray images using the DenseNet121 architecture. The model is trained on a dataset of chest X-rays and evaluated for its ability to classify images as either pneumonia or healthy.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Training the Model](#training-the-model)
7. [Evaluation](#evaluation)
8. [License](#license)

## Overview

This project leverages the DenseNet121 model, pre-trained on ImageNet, as the backbone for pneumonia classification. The code includes image preprocessing, augmentation, training, and evaluation on a dataset of chest X-ray images. The final model is saved and can be used for inference on new images.

## Installation

To get started, you'll need to have Python and the necessary libraries installed. The following libraries are required:

- TensorFlow
- NumPy
- tqdm

You can install the necessary dependencies using pip:

```bash
pip install tensorflow numpy tqdm
```

## Dataset

The model is trained on a chest X-ray dataset, which can be organized into the following directory structure:

```
/chest_xray
    /train
        /NORMAL
        /PNEUMONIA
    /val
        /NORMAL
        /PNEUMONIA
    /test
        /NORMAL
        /PNEUMONIA
```

- The **train** directory contains training images, divided into subdirectories for normal and pneumonia images.
- The **val** directory contains validation images, also divided into subdirectories.
- The **test** directory contains test images.

Ensure that you download and organize the dataset into this structure before running the code.

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/your-repository/pneumonia-detection.git
cd pneumonia-detection
```

2. Update the path to your dataset in the code:

```python
base_dir = r"C:\path\to\your\dataset\chest_xray"
```

3. Run the script to start training the model:

```bash
python train_model.py
```

The model will train for 5 epochs and output training/validation metrics during each epoch.

## Model Architecture

The model uses DenseNet121 as the base model, with the final layer replaced by a softmax classifier for binary classification (pneumonia vs. normal):

1. **DenseNet121**: Pre-trained on ImageNet, used for feature extraction.
2. **GlobalAveragePooling2D**: Reduces spatial dimensions, outputting a single vector.
3. **Dense**: Final output layer with 2 units (normal vs pneumonia) and softmax activation.

## Training the Model

The model is trained using the following parameters:

- **Optimizer**: Adam with a learning rate of 0.001
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Image Size**: 224x224 pixels

The training loop will run for a specified number of epochs (default 5), printing out loss and accuracy metrics for both training and validation.

## Evaluation

After training, the model is saved as `pneumonia_densenet121.h5`. The model is then evaluated on the test set, and the test accuracy is printed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed by Muhammad Rasoul Sahibzadah
