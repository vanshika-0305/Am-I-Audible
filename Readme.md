# Pharaoh Vision

## Overview

This project provides a machine learning pipeline for training and inference using predefined datasets. The models can be used directly with the provided pipeline or trained from scratch using the datasets.\
The main aim of this project is to understand the usage of Deep CNN Image Classification techniques for Audio files.

## Dataset

To use this project, download the dataset from the following link: [Dataset Download Link](https://drive.google.com/drive/folders/1-XSXm3zo0Tt-eTJF68_03fw41FFORbvV?usp=share_link)

 After downloading, extract the dataset and place it in the `Data/` directory within the project folder or just copy the path from the downloads folder and use it where mentioned.

## Installation

Ensure you have Python installed (preferably version 3.x). Then, install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Running Inference

If you want to use the pre-trained model, run the following command:

```bash
python app.py 
```

This will open up a web page where you can upload files to get predictions.

### Training the Model

If you want to train the model on the dataset yourself (just copy the dataset path you downloaded and paste it), run:

```bash
python train_bird_model.py --data_path <Your Dataset Path>
```

This will initiate training using the provided dataset.

## Models

### 1. **Speaker Classification Model**

This model classifies human audio files from the **VoxCeleb Indian dataset (WAV files)** into speaker categories. It utilizes **VGG16** as a feature extractor and a custom **DNN** for classification. Hyperparameters such as dense layer units, dropout rates, and learning rate are optimized using **Keras Tuner**.

### 2. **Siamese Network for Speaker Verification**

A Siamese network is used for speaker verification. Two loss functions were explored:

- **Triplet Loss**: Trained on **500 samples** due to limited compute power, achieving **60% accuracy**.
- **Contrastive Loss**: Achieved **85% accuracy** on the training set and **77% accuracy** on the testing set, minimizing loss to **0.084**.

#### Preprocessing for Siamese Network

- Extracting **Mel Spectrograms**.
- Resampling, padding, truncating audio.
- Creating **positive and negative pairs** for training.

### 3. **Bird Species Classification Model**

This model classifies different bird species with high accuracy:

- **95% accuracy on the training set**.
- **87% accuracy on the validation set**.
- Uses a **Softmax output layer** to provide a probability distribution for classification.
- A fully functional **frontend** is integrated for user interaction.

## Pipeline for Bird Model

1. **Data Preprocessing**: The dataset is cleaned and transformed before training.
2. **Model Training**: The model is trained on the dataset, with hyperparameters defined in `config.py`.
3. **Evaluation**: The trained model is evaluated using predefined metrics.
4. **Inference**: The trained model is used to make predictions on new input data.

## Configuration

Modify `config.py` to adjust hyperparameters and training settings.

## Contributing

Feel free to contribute by creating pull requests or raising issues.

