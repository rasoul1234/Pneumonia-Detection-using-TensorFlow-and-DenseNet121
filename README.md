# Pneumonia Detection using TensorFlow and DenseNet121

This project is a Pneumonia detection system built using TensorFlow/Keras and DenseNet121, trained on chest X-ray images. The system uses transfer learning to classify X-ray images into two categories: Pneumonia and Normal.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Saving the Model](#saving-the-model)
- [Results](#results)

## Project Structure
```
Pneumonia_Detector/
├── chest_xray-Dataset/       # Dataset directory
│   ├── train/              # Training images
│   ├── val/                # Validation images
│   └── test/               # Testing images
├── main.py                  # Main Python script for training and evaluation
└── pneumonia_densenet121.h5 # Saved model
```

## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.8+
- TensorFlow 2.x
- NumPy
- tqdm

You can install the dependencies using pip:
```bash
pip install tensorflow numpy tqdm
```

## Dataset
The dataset used for this project is the **Chest X-ray Dataset**, which is organized into three directories:

- `train`: Contains training images.
- `val`: Contains validation images.
- `test`: Contains testing images.

Each directory contains subdirectories for two classes:
- `NORMAL`
- `PNEUMONIA`

Ensure the dataset structure matches the one described in the [Project Structure](#project-structure) section.

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/your-repo/pneumonia-detector.git
cd pneumonia-detector
```

2. Place the dataset in the `chest_xray-Dataset` folder as described above.

3. Run the `main.py` script:
```bash
python main.py
```

This will train the DenseNet121 model for 5 epochs and save the trained model as `pneumonia_densenet121.h5`.

## Model Architecture
The model is based on the **DenseNet121** architecture from TensorFlow's Keras Applications module. Key features include:

- **Pretrained weights**: Initialized with ImageNet weights.
- **Global Average Pooling**: Added after the feature extraction layers.
- **Fully Connected Layer**: A dense layer with 2 output units for binary classification (Normal and Pneumonia).
- **Activation Function**: Softmax for classification.

## Training and Evaluation
### Training
- **Data Augmentation**: Performed on the training dataset using horizontal flips and normalization.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Loss Function**: Categorical Crossentropy.
- **Batch Size**: 32.
- **Epochs**: 5 (configurable).

### Evaluation
The model is evaluated on the test set using accuracy as the primary metric.

### Key Functions:
- `train_model`: Handles the training and validation process.
- `evaluate_model`: Evaluates the model on the test dataset and prints accuracy.

## Saving the Model
After training, the model is saved as `pneumonia_densenet121.h5` for future inference or fine-tuning.

## Results
During training, the script displays the following metrics:
- Loss
- Accuracy
- Validation Loss
- Validation Accuracy

These metrics are displayed for each epoch, providing a detailed view of the model's performance.

After evaluation, the test accuracy is displayed, giving a clear indication of the model's generalization ability.

---

Feel free to extend the project by:
- Experimenting with different architectures.
- Adding more data augmentation techniques.
- Fine-tuning hyperparameters like the learning rate or batch size.

Developed by Muhammad Rasoul Sahibzadah

