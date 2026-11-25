import numpy as np
from scipy.fft import rfft, rfftfreq
from typing import Union
import glob


def extract_dominant_frequency_and_max_amplitude(data_input:Union[np.ndarray,str], fs=200):
    """
    Extract for each channel of EEG data every 1 second:
    1. The dominant frequency of the band with the highest proportion
    2. Half of the peak-to-peak value in that time segment

    Args:
        data_input (Union[ndarray,str]): EEG data file path (.npy format) or data array (shape [channels, samples])
        fs (int): Sampling rate (Hz), default is 200

    Returns:
        tuple: Tuple containing two arrays
            - dominant_freq_matrix (ndarray): Dominant frequency matrix (channels × segments)
            - max_amp_matrix (ndarray): Half peak-to-peak value matrix (channels × segments)
    """
    # Define frequency band ranges (embedded directly in function)
    BANDS = {
        'δ (0.3-3.5Hz)': (0.3, 3.5),
        'θ (4-7.5Hz)': (4, 7.5),
        'α (8-13Hz)': (8, 13),
        'β (14-30Hz)': (14, 30),
        'γ (30-70Hz)': (30, 70)
    }
    
    # Load data according to input type
    if isinstance(data_input, str):
        data = np.load(data_input)
    else:
        data = data_input
    
    n_channels, total_samples = data.shape
    segment_duration = 1.0
    segment_samples = int(fs * segment_duration)
    n_segments = total_samples // segment_samples

    dominant_freq_matrix = np.zeros((n_channels, n_segments))
    max_amp_matrix = np.zeros((n_channels, n_segments))

    for channel in range(n_channels):
        for seg_idx in range(n_segments):
            start_idx = seg_idx * segment_samples
            end_idx = start_idx + segment_samples
            segment_data = data[channel, start_idx:end_idx]

            # Calculate half of the peak-to-peak value
            pp = np.max(segment_data) - np.min(segment_data)  # Peak-to-peak value
            half_pp = pp / 2.0  # Half of peak-to-peak value
            max_amp_matrix[channel, seg_idx] = half_pp

            # FFT calculation
            yf = rfft(segment_data)
            xf = rfftfreq(len(segment_data), 1/fs)
            spectrum = np.abs(yf)

            total_energy = np.sum(spectrum)
            if total_energy == 0:
                dominant_freq_matrix[channel, seg_idx] = 0
                continue

            band_percentages = {}
            for band, (low, high) in BANDS.items():
                idx = np.where((xf >= low) & (xf <= high))[0]
                if len(idx) > 0:
                    band_energy = np.sum(spectrum[idx])
                    band_percentages[band] = band_energy / total_energy
                else:
                    band_percentages[band] = 0

            dominant_band = max(band_percentages, key=band_percentages.get)
            low, high = BANDS[dominant_band]
            band_mask = (xf >= low) & (xf <= high)
            if np.any(band_mask):
                band_spectrum = spectrum[band_mask]
                band_freqs = xf[band_mask]
                peak_in_band_idx = np.argmax(band_spectrum)
                dominant_freq = band_freqs[peak_in_band_idx]
            else:
                peak_idx = np.argmax(spectrum)
                dominant_freq = xf[peak_idx]

            dominant_freq_matrix[channel, seg_idx] = dominant_freq
    
    return dominant_freq_matrix, max_amp_matrix


if __name__ == '__main__':
    file = glob.glob('../npy_data/*.npy')
    dominant_freq_matrix, max_amp_matrix = extract_dominant_frequency_and_max_amplitude(file[0])
    print(dominant_freq_matrix)
    print(max_amp_matrix)