import os
from tqdm import tqdm
from utils.preprocessing import split_audio, save_spectrogram

def user_preprocessing_bird(input_dir):
    try:
        output_dir = os.path.join(os.getcwd(), "birdclipswav")
    except Exception as e:
        print(f"Error: {e}")
        print("Put the data files in the correct folder!!")
        return

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of bird species (folders)
    bird_species = os.listdir(input_dir)

    print("Processing bird audio files...")

    for bird in tqdm(bird_species, desc="Splitting audio", unit="bird"):
        bird_folder = os.path.join(input_dir, bird)
        if not os.path.isdir(bird_folder):
            continue  # Skip if it's not a directory

        wavfiles = os.listdir(bird_folder)
        if len(wavfiles) < 10:  # Ignore birds with less data
            continue

        for wavfile in tqdm(wavfiles, desc=f"Processing {bird}", unit="file", leave=False):
            if wavfile.endswith(".mp3"):
                input_path = os.path.join(bird_folder, wavfile)
                output_path = os.path.join(output_dir, bird)
                os.makedirs(output_path, exist_ok=True)
                split_audio(input_path, output_path)

    input_root = os.path.join(os.getcwd(), 'birdclipswav')
    output_root = os.path.join(os.getcwd(), 'birds_spectrograms')

    print("\nGenerating spectrograms...")

    # Walk through the directory structure
    for root, _, files in tqdm(os.walk(input_root), desc="Processing folders", unit="folder"):
        if len(files) < 10:
            continue

        for file in tqdm(files, desc=f"Generating spectrograms in {os.path.basename(root)}", unit="spectrogram", leave=False):
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_root)  # Preserve directory structure
                output_dir = os.path.join(output_root, relative_path)
                os.makedirs(output_dir, exist_ok=True)  # Create output directory

                png_filename = os.path.splitext(file)[0] + ".png"
                png_path = os.path.join(output_dir, png_filename)

                if not os.path.exists(png_path):  # Check if PNG already exists
                    save_spectrogram(wav_path, png_path)

    print("\n✅ Spectrogram extraction complete!")

