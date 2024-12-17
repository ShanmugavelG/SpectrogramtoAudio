
import argparse
from pathlib import Path
import cv2
import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def list_files(source):
    path = Path(source)
    files = [file for file in path.rglob('*') if file.is_file()]
    return files


def audio_to_spectrogram(audio_path, save_path, duration, n_fft=2048, hop_length=512, win_length=None):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, duration=duration)

        # Compute spectrogram with higher resolution
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        S = librosa.amplitude_to_db(abs(D), ref=np.max)

        # Normalize values to 0-255 range and convert to uint8 for better visual quality
        S = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert to RGB and save as PNG
        S = cv2.cvtColor(S, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(save_path, S)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")


def process_file(file, source, output, duration):
    """
    Process a single file by converting it to a spectrogram.

    Args:
        file (Path): Path object representing the file to process.
        source (str): The source directory path.
        output (str): The output directory path.
        duration (int): Duration of the audio file to process in seconds.

    Returns:
        None
    """
    # Output path
    new_path = Path(str(file).replace(str(source), output))

    # Create output directory
    new_path.parent.mkdir(parents=True, exist_ok=True)

    # Replace suffix
    new_path = new_path.with_suffix('.png')

    # Convert
    audio_to_spectrogram(str(file), str(new_path), duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cats_dogs', help='source folder')
    parser.add_argument('--duration', type=int, default=200, help='duration of audios in case they are too big')
    parser.add_argument('--output', type=str, default='output', help='folder output')
    opt = parser.parse_args()
    source, duration, output = opt.source, opt.duration, opt.output

    file_list = list_files(source)

    # Use multiprocessing to speed up processing
    with ProcessPoolExecutor() as executor:
        for file in tqdm(file_list):
            executor.submit(process_file, file, source, output, duration)
