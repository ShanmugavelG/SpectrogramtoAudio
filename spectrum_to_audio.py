import argparse
from pathlib import Path
import cv2
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.signal import butter, filtfilt


def list_files(source):
    """
    List all files in the given source directory and its subdirectories.

    Args:
        source (str): The source directory path.

    Returns:
        list: A list of Path objects representing the files.
    """
    path = Path(source)
    files = [file for file in path.rglob('*.png') if file.is_file()]
    return files


def butter_lowpass(cutoff, fs, order=5):
    """
    Design a Butterworth low-pass filter.

    Args:
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling rate of the signal.
        order (int): The order of the filter.

    Returns:
        b, a: The filter coefficients.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_lowpass_filter(data, cutoff, fs, order=1):
    """
    Apply a Butterworth low-pass filter to a signal.

    Args:
        data (np.array): The input signal.
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling rate of the signal.
        order (int): The order of the filter.

    Returns:
        y (np.array): The filtered signal.
    """
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y


def spectral_subtraction(S, noise_estimate):
    """
    Apply spectral subtraction to reduce noise.

    Args:
        S (np.array): The spectrogram.
        noise_estimate (np.array): Estimated noise spectrum.

    Returns:
        np.array: The cleaned spectrogram.
    """
    # Subtract noise estimate from spectrogram, ensuring no negative values
    S_cleaned = np.maximum(S - noise_estimate[:, np.newaxis], 0)
    return S_cleaned


def estimate_noise(S, noise_floor_percent=0.05):
    """
    Estimate the noise from the spectrogram by averaging the lowest percent of frames.

    Args:
        S (np.array): The spectrogram.
        noise_floor_percent (float): Percentage of frames to use for noise estimation.

    Returns:
        np.array: The estimated noise spectrum.
    """
    num_frames = S.shape[1]
    num_noise_frames = max(1, int(noise_floor_percent * num_frames))
    noise_estimate = np.mean(np.sort(S, axis=1)[:, :num_noise_frames], axis=1)
    return noise_estimate


def spectrogram_to_audio(spectrogram_path, save_path, sr=22050, n_fft=4096, hop_length=1024, num_iters=200):
    """
    Convert a spectrogram image back to an audio file with enhanced quality.

    Args:
        spectrogram_path (str): The path to the spectrogram image.
        save_path (str): The path to save the reconstructed audio.
        sr (int): Sample rate of the audio.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.
        num_iters (int): Number of iterations for the Griffin-Lim algorithm.

    Returns:
        None
    """
    # Load the spectrogram image
    S_img = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)

    # Convert image to float32 and scale back to original amplitude range
    S = cv2.normalize(S_img, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    
    # Convert back to amplitude
    S = librosa.db_to_amplitude(S * 80.0 - 80.0)

    # Estimate noise and perform spectral subtraction
    noise_estimate = estimate_noise(S, noise_floor_percent=0.05)
    S_cleaned = spectral_subtraction(S, noise_estimate)

    # Use Griffin-Lim algorithm for phase reconstruction
    y = librosa.griffinlim(S_cleaned, n_iter=num_iters, hop_length=hop_length, win_length=n_fft)

    # Apply a low-pass filter to smooth out any harsh frequencies
    y = apply_lowpass_filter(y, cutoff=sr // 4, fs=sr)

    # Save the recovered audio to a file using soundfile
    sf.write(save_path, y, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='output', help='source folder containing spectrogram images')
    parser.add_argument('--output', type=str, default='reconstructed_audio', help='folder to save reconstructed audio files')
    parser.add_argument('--sample_rate', type=int, default=22050, help='sample rate for the audio file')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512, help='Number of samples between successive frames')
    parser.add_argument('--num_iters', type=int, default=200, help='Number of iterations for the Griffin-Lim algorithm')
    opt = parser.parse_args()

    source = opt.source
    output = opt.output
    sr = opt.sample_rate
    n_fft = opt.n_fft
    hop_length = opt.hop_length
    num_iters = opt.num_iters

    # Create output directory if it doesn't exist
    Path(output).mkdir(parents=True, exist_ok=True)

    file_list = list_files(source)

    for file in tqdm(file_list):
        # Output path
        new_path = Path(str(file).replace(source, output))
        new_path = new_path.with_suffix('.wav')

        # Convert
        spectrogram_to_audio(str(file), str(new_path), sr=sr, n_fft=n_fft, hop_length=hop_length, num_iters=num_iters)
