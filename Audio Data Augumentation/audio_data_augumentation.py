import soundfile as sf
import numpy as np

def add_environmental_noise(signal, desired_snr_db):
    # Calculate the power of the signal
    signal_power = np.mean(signal ** 2)
    # Calculate the required noise power for the desired SNR
    snr_linear = 10 ** (desired_snr_db / 10)
    noise_power = signal_power / snr_linear
    # Generate synthetic environmental noise
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    # Add the noise to the signal
    noisy_signal = signal + noise
    # Clip the noisy signal to ensure it stays within the -1.0 to 1.0 range
    noisy_signal = np.clip(noisy_signal, -1.0, 1.0)

    return noisy_signal

def add_artifact_noise(signal, desired_snr_db):
    # Calculate the power of the signal
    signal_power = np.mean(signal ** 2)
    # Calculate the required noise power for the desired SNR
    snr_linear = 10 ** (desired_snr_db / 10)
    noise_power = signal_power / snr_linear
    # Generate base artificial noise
    artifact_noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    # Adding sporadic intense noise spikes at regular intervals
    for i in range(len(artifact_noise)):
        if i % 50 == 0:  # Example: Simulate stethoscope movement every 50 samples
            artifact_noise[i] += np.random.normal(0, np.sqrt(noise_power))
    # Add the artificial noise to the signal
    noisy_signal = signal + artifact_noise
    # Clip the noisy signal to ensure it stays within the -1.0 to 1.0 range
    noisy_signal = np.clip(noisy_signal, -1.0, 1.0)
    return noisy_signal

def add_respiratory_noise(signal, desired_snr_db, sampling_rate):
    # Calculate the power of the signal
    signal_power = np.mean(signal ** 2)
    # Calculate the required noise power for the desired SNR
    snr_linear = 10 ** (desired_snr_db / 10)
    noise_power = signal_power / snr_linear
    # Create a time array corresponding to the signal
    time = np.arange(len(signal)) / sampling_rate
    # Generate respiratory noise (sinusoidal wave)
    respiratory_noise = np.sin(2 * np.pi * 0.2 * time)  # Frequency of 0.2 Hz for respiratory cycle
    # Scale the respiratory noise to achieve the desired noise power
    respiratory_noise = respiratory_noise * np.sqrt(noise_power / np.mean(respiratory_noise ** 2))
    # Add the scaled respiratory noise to the signal
    noisy_signal = signal + respiratory_noise
    # Clip the noisy signal to ensure it stays within the -1.0 to 1.0 range
    noisy_signal = np.clip(noisy_signal, -1.0, 1.0)
    return noisy_signal


file_path = "address path of heartbeat audio file"
signal, sr = sf.read(file_path)

# Choosen db are -3, -2, 2, 3, 5
desired_snr_db = 5 

# Combine all types of noise
noisy_pcg_all_noises = signal + \
    add_environmental_noise(signal, desired_snr_db) + \
    add_artifact_noise(np.zeros_like(signal), desired_snr_db) +\
    add_respiratory_noise(np.zeros_like(signal), desired_snr_db, sr)
