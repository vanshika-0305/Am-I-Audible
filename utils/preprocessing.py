import pandas as pd
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import os 
from pydub import AudioSegment

def split_audio(input_path, output_path, clip_length=5000):
    """
    Splits an audio file into smaller clips of a specified length and saves them as WAV files.

    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Directory where the split audio clips will be saved.
        clip_length (int, optional): Length of each clip in milliseconds. Default is 5000 ms (5 seconds).
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_path)
    
    # Calculate the number of clips
    total_length = len(audio)
    num_clips = total_length // clip_length
    
    
    # Split the audio file into clips
    for i in range(num_clips + 1):
        start_time = i * clip_length
        end_time = start_time + clip_length
        clip = audio[start_time:end_time]
        
        # Save the clip as WAV
        clip_filename = os.path.join(output_path, f"{os.path.basename(input_path).split('.')[0]}clip_{i + 1}.wav")
        clip.export(clip_filename, format="wav")

# input_root = os.path.join(os.getcwd(), 'birdclipswav')
# output_root = os.path.join(os.getcwd(), 'birds_spectrograms')

# os.makedirs(output_root, exist_ok=True)

# Function to generate spectrogram
def save_spectrogram(wav_path, png_path):
    """
    Generates and saves a spectrogram from a given WAV audio file.

    Args:
        wav_path (str): Path to the input WAV file.
        png_path (str): Path where the generated spectrogram image will be saved.
    """
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Save the spectrogram as PNG
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close('all')  # Close the figure to free up memory
    del y, S, S_db  # Clear variables to free up memory

# Walk through the directory structure
def process_audio_files(input_root, output_root, min_files=10):
    """
    Processes audio files in a directory, generating spectrograms for each .wav file.
    
    Args:
        input_root (str): Path to the directory containing .wav files.
        output_root (str): Path to the directory where spectrograms will be saved.
        min_files (int): Minimum number of files in a folder to be processed.
    """
    for root, _, files in os.walk(input_root):
        if len(files) < min_files:
            continue  # Skip directories with fewer files than min_files
        
        for file in files:
            if file.endswith(".wav"):
                try:
                    wav_path = os.path.join(root, file)
                    duration = librosa.get_duration(path=wav_path)
                    if duration < 4.0:
                        print(f"Skipping {file}: too short ({duration:.2f}s)")
                        continue  # Skip short files
                    relative_path = os.path.relpath(root, input_root)  # Preserve directory structure
                    output_dir = os.path.join(output_root, relative_path)
                    os.makedirs(output_dir, exist_ok=True)  # Create output directory

                    png_filename = os.path.splitext(file)[0] + ".png"
                    png_path = os.path.join(output_dir, png_filename)

                    if not os.path.exists(png_path):  # Check if PNG already exists
                        save_spectrogram(wav_path, png_path)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

