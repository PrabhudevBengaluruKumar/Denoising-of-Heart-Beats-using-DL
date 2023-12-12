import numpy as np

def estimate_snr(original, enhanced):
    # Make the signals the same length
    min_length = min(len(original), len(enhanced))
    original = original[:min_length]
    enhanced = enhanced[:min_length]
    
    noise = original - enhanced
    noise_power = np.mean(noise**2) if np.mean(noise**2) > 0 else 1e-7
    signal_power = np.mean(enhanced**2) if np.mean(enhanced**2) > 0 else 1e-7
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

signal = "raw signal"
combined_signal = "denoised signal"
model_processed_snr = estimate_snr(signal, combined_signal)