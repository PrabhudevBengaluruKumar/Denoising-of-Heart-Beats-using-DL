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

# def add_artifact_noise(signal, noise_level):
#     artifact_noise = np.random.normal(0, noise_level, len(signal))
#     for i in range(len(artifact_noise)):
#         if i % 50 == 0:  # Simulate stethoscope movement every 50 samples
#             artifact_noise[i] += np.random.normal(0, noise_level)
#     noisy_signal = signal + artifact_noise
#     return noisy_signal

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


# def add_respiratory_noise(signal, noise_level, sampling_rate):
#     time = np.arange(len(signal)) / sampling_rate
#     respiratory_noise = 0.1 * np.sin(2 * np.pi * 0.2 * time)  # Simulate respiratory noise
#     noisy_signal = signal + noise_level * respiratory_noise
#     return noisy_signal

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

def add_electronic_interference(signal, desired_snr_db, sampling_rate, interference_freq):
    # Calculate the power of the signal
    signal_power = np.mean(signal ** 2)
    # Calculate the required noise power for the desired SNR
    snr_linear = 10 ** (desired_snr_db / 10)
    noise_power = signal_power / snr_linear
    # Create a time array corresponding to the signal
    time = np.arange(len(signal)) / sampling_rate
    # Generate sinusoidal interference
    sinusoidal_interference = np.sin(2 * np.pi * interference_freq * time)
    # Create Gaussian noise
    gaussian_noise = np.random.normal(0, 1, len(signal))
    # Combine sinusoidal interference and Gaussian noise
    combined_interference = sinusoidal_interference + gaussian_noise
    # Scale the combined interference to achieve the desired noise power
    combined_interference = combined_interference * np.sqrt(noise_power / np.mean(combined_interference ** 2))
    # Add the scaled interference to the signal
    noisy_signal = signal + combined_interference
    # Clip the noisy signal to ensure it stays within the -1.0 to 1.0 range
    noisy_signal = np.clip(noisy_signal, -1.0, 1.0)
    return noisy_signal

def add_cable_movement_noise(signal, desired_snr_db, disturbance_frequency, disturbance_duration, sampling_rate):
    # Calculate the power of the signal
    signal_power = np.mean(signal ** 2)
    # Convert desired SNR from dB to linear scale
    desired_snr_linear = 10 ** (desired_snr_db / 10)
    # Calculate the power of the noise based on the desired SNR
    noise_power = signal_power / desired_snr_linear
    # Initialize the cable movement noise array
    cable_movement_noise = np.zeros(len(signal))
    # Calculate the interval in samples for disturbances based on frequency
    disturbance_interval = int(sampling_rate / disturbance_frequency)
    # Iterate over the signal to add disturbances
    for i in range(0, len(signal), disturbance_interval):
        if np.random.rand() < 0.5:  # Probability of disturbance occurrence
            duration_samples = int(disturbance_duration * sampling_rate)
            end_index = min(i + duration_samples, len(signal))

            # Generate random noise for the duration of the disturbance
            disturbance_noise = np.random.normal(0, np.sqrt(noise_power), end_index - i)
            cable_movement_noise[i:end_index] = disturbance_noise
    # Scale the entire noise array to achieve the desired noise power
    if np.any(cable_movement_noise != 0):
        cable_movement_noise = cable_movement_noise * np.sqrt(noise_power / np.mean(cable_movement_noise ** 2))
    # Add the cable movement noise to the original signal
    noisy_signal = signal + cable_movement_noise
    # Ensure the signal stays within bounds
    noisy_signal = np.clip(noisy_signal, -1.0, 1.0)
    return noisy_signal



def all_5_noises(signal, desired_snr_db, interference_freq, disturbance_frequency, disturbance_duration, sampling_rate):

    signal_power = np.mean(signal ** 2)
    desired_snr_linear = 10 ** (desired_snr_db / 10)
    noise_power = signal_power / desired_snr_linear
    time = np.arange(len(signal)) / sampling_rate

    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    artifact_noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    for i in range(len(artifact_noise)):
        if i % 50 == 0: 
            artifact_noise[i] += np.random.normal(0, np.sqrt(noise_power))

    respiratory_noise = np.sin(2 * np.pi * 0.2 * time)  
    respiratory_noise = respiratory_noise * np.sqrt(noise_power / np.mean(respiratory_noise ** 2))

    sinusoidal_interference = np.sin(2 * np.pi * interference_freq * time)
    gaussian_noise = np.random.normal(0, 1, len(signal))
    combined_interference = sinusoidal_interference + gaussian_noise
    combined_interference = combined_interference * np.sqrt(noise_power / np.mean(combined_interference ** 2))

    cable_movement_noise = np.zeros(len(signal))
    disturbance_interval = int(sampling_rate / disturbance_frequency)
    for i in range(0, len(signal), disturbance_interval):
        if np.random.rand() < 0.5:  # Probability of disturbance occurrence
            duration_samples = int(disturbance_duration * sampling_rate)
            end_index = min(i + duration_samples, len(signal))
            disturbance_noise = np.random.normal(0, np.sqrt(noise_power), end_index - i)
            cable_movement_noise[i:end_index] = disturbance_noise
    if np.any(cable_movement_noise != 0):
        cable_movement_noise = cable_movement_noise * np.sqrt(noise_power / np.mean(cable_movement_noise ** 2))

    noisy_signal = signal + noise + artifact_noise + respiratory_noise + combined_interference + cable_movement_noise 
    noisy_signal = np.clip(noisy_signal, -1.0, 1.0)
    return noisy_signal