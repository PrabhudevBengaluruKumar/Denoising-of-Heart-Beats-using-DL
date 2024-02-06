import numpy as np

# def estimate_snr(original, enhanced):
#     # Make the signals the same length
#     min_length = min(len(original), len(enhanced))
#     original = original[:min_length]
#     enhanced = enhanced[:min_length]
    
#     noise = original - enhanced
#     noise_power = np.mean(noise**2) if np.mean(noise**2) > 0 else 1e-7
#     signal_power = np.mean(enhanced**2) if np.mean(enhanced**2) > 0 else 1e-7
    
#     snr = 10 * np.log10(signal_power / noise_power)
#     return snr

def estimate_snr(original, enhanced):
    # Ensure signals are the same length for fair comparison
    min_length = min(len(original), len(enhanced))
    original = original[:min_length]
    enhanced = enhanced[:min_length]
    # Calculate the difference as noise
    noise = original - enhanced
    noise_power = np.mean(noise**2) if np.mean(noise**2) > 0 else 1e-7
    # Consider using the original signal's power for a potentially more accurate representation
    original_signal_power = np.mean(original**2) if np.mean(original**2) > 0 else 1e-7
    # Calculate SNR
    snr = 10 * np.log10(original_signal_power / noise_power)
    return snr

signal = "raw signal"
combined_signal = "denoised signal"
model_processed_snr = estimate_snr(signal, combined_signal)
