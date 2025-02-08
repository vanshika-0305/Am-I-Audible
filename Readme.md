# Pharaoh Vision

## Overview
This project provides a machine learning pipeline for training and inference using a predefined dataset. The model can be used directly with the provided pipeline or trained from scratch using the dataset.

## Dataset
To use this project, download the dataset from the following link:
[Dataset Download Link](<insert_link_here>)

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
This will open up a web page where you can upload file to get predictions.

### Training the Model
If you want to train the model on the dataset yourself (just copy dataset path you downloaded and paste it), run:
```bash
python train_bird_model.py --data_path <Your Dataset Path>
```
This will initiate training using.

## Pipeline
1. **Data Preprocessing**: The dataset is cleaned and transformed before training.
2. **Model Training**: The model is trained on the dataset, with hyperparameters defined in `config.py`.
3. **Evaluation**: The trained model is evaluated using predefined metrics.
4. **Inference**: The trained model is used to make predictions on new input data.

## Configuration
Modify `config.py` to adjust hyperparameters and training settings.

## Contributing
Feel free to contribute by creating pull requests or raising issues.

