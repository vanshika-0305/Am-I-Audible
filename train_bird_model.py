import datetime
import argparse
import os
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Rescaling
from keras.models import Sequential
from keras.utils import image_dataset_from_directory
from keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback  # Import tqdm callback
from utils.user_preprocessing_bird import user_preprocessing_bird
from utils.config import Parameters

parameters = Parameters()

# Argument parser for command-line input
parser = argparse.ArgumentParser(description="Train a CNN model on bird spectrograms")
parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
args = parser.parse_args()

# for preprocessing
user_preprocessing_bird(os.path.join(args.data_path, "Voice of Birds/Voice of Birds"))

# Early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Load dataset
train_ds = image_dataset_from_directory(
    os.path.join(os.getcwd(), "birds_spectrograms"),
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=parameters.batch_size,
    image_size=(128, 128),
    seed=123,
    validation_split=0.2,
    subset='training')

validation_ds = image_dataset_from_directory(
    os.path.join(os.getcwd(), "birds_spectrograms"),
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=parameters.batch_size,
    image_size=(128, 128),
    seed=123,
    validation_split=0.2,
    subset='validation')

normalization_layer = Rescaling(1./255)  # Normalize to (0-1)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
validation_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))

# Define the CNN model
model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(84, activation='softmax')
])

# Compile the model
model.compile(optimizer=parameters.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with tqdm progress bar
model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=parameters.epochs,
    callbacks=[early_stopping, TqdmCallback()]
)

# Save the model
time = datetime.datetime.now()
time.strftime("%Y-%m-%d-%H:%M:%S")
model.save(f"birds_model.h5{time}")

print(f"Training complete. Model saved as birds_model.h5{time} 🎉")
