import os
import tensorflow as tf
from keras.models import load_model
from utils import preprocessing
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# classes = os.listdir("birds_spectrograms")

def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, color_mode="grayscale", target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize same as training
    return img_array

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
        os.path.join(spectrograms, f"{time}"),
        min_files=0
    )
    prediction = []
    for root, _, files in os.walk(spectrograms):
        for file in files:
            print(os.path.join(root,file))
            image = preprocess_image(os.path.join(root,file))
            prediction.append(np.argmax(model.predict(image)))
    
    return prediction
