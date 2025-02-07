import os
import tensorflow as tf
from keras.models import load_model
from utils import preprocessing
from keras.models import load_model
import cv2
import numpy as np

# classes = os.listdir("birds_spectrograms")

def read_image(path):
    image = cv2.imread(path)
    if image is None:
        print("Error: Image not found or failed to load!")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 128), interpolation = cv2.INTER_LINEAR)
        image = image.reshape((1, 128, 128, 1)) / 255
    return image

def predict(model, time, file):
    root = os.getcwd()
    spectrograms = os.path.join(root, "UserData", "Spectrogram")
    output_path = os.path.join(root, f"UserData/{time}")
    os.makedirs(output_path, exist_ok=True)
    preprocessing.split_audio(
        os.path.join(root, "UserData", "Soundfile", file),
        os.path.join(root, output_path)
    )
    os.makedirs(os.path.join(spectrograms, f"{time}"), exist_ok=True)
    preprocessing.process_audio_files(
        os.path.join(root, f"UserData/{time}"),
        os.path.join(spectrograms, f"{time}")
    )
    prediction = []
    for root, _, files in os.walk(spectrograms):
        for file in files:
            print(os.path.join(root,file))
            image = read_image(os.path.join(root,file))
            prediction.append(np.argmax(model.predict(image)))
    
    return prediction
