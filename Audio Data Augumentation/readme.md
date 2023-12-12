# Heartbeat Audio Signal Noise Addition Script

This script is designed to simulate various types of noise in a heartbeat audio file, creating a more realistic environment for testing and analysis. It adds environmental noise, artifact noise, and respiratory noise to the original signal.

## Dependencies
- `numpy`: For numerical operations and signal processing.
- `soundfile`: For reading and writing audio files.

## Functions
### `add_environmental_noise(signal, desired_snr_db)`
- Adds random environmental noise to the signal.
- `signal`: The original audio signal (array).
- `desired_snr_db`: The desired signal-to-noise ratio in decibels.
### `add_artifact_noise(signal, desired_snr_db)`
- Adds artificial noise with sporadic intense noise spikes, simulating things like stethoscope movement or other intermittent artifacts.
- Noise spikes are added at regular intervals.
### `add_respiratory_noise(signal, desired_snr_db, sampling_rate)`
- Simulates respiratory noise as a sinusoidal wave.
- The frequency of the wave is set to simulate a typical respiratory cycle (0.2 Hz).
- `sampling_rate`: The sampling rate of the audio signal.

## Usage
1. **Load the Audio File**
   - Update `file_path` with the path of the heartbeat audio file.
   - The script reads the file and extracts the signal and sampling rate.
2. **Set Desired SNR**
   - `desired_snr_db`: Set the desired signal-to-noise ratio for the noise addition.
3. **Combine All Types of Noise**
   - The script combines environmental, artifact, and respiratory noises with the original signal.
4. **Output**
   - The final output is `noisy_pcg_all_noises`, which is the original signal combined with all types of noise.

## Note
- Ensure that the audio file format is compatible with the `soundfile` library.
- The soundfile library normalises the audio within the -1.0 to 1.0 range.
- The `np.clip` method is used in all noise addition functions to ensure that the noisy signal stays within the -1.0 to 1.0 range, avoiding distortion. (Here clipping maintained same as normalisation done by soundfile library).
- The script is designed for testing and research purposes, helping to understand how different types of noise affect heartbeat audio analysis.

## Example
```python
file_path = "path/to/heartbeat/audio/file.wav"
desired_snr_db = 5  # Set the desired SNR in dB
signal, sr = sf.read(file_path)
noisy_pcg_all_noises = signal + \
    add_environmental_noise(signal, desired_snr_db) + \
    add_artifact_noise(np.zeros_like(signal), 5) +\
    add_respiratory_noise(np.zeros_like(signal), 5, sr)
# Now `noisy_pcg_all_noises` contains the signal with all noises added
```

# Explanation of noise creation 

## 1. Artificial noise
In real-world scenarios, environmental noise often consists of random sounds from various sources (like traffic, wind, people talking, etc.), which can be roughly modeled by a Gaussian distribution. The Gaussian noise does not replicate specific sounds but rather the random and unpredictable nature of real-world environmental noise.
### Calculate the Power of the Signal
The power of the signal is computed as the mean of the squared signal values. This measure provides an estimation of the signal's 'loudness' or energy.
### Determine Required Noise Power for Desired SNR
- SNR (Signal-to-Noise Ratio) quantifies the level of a desired signal to the level of background noise.
- The SNR is provided in decibels (dB) and converted into a linear scale using `10 ** (desired_snr_db / 10)`.
- The noise power is then calculated by dividing the signal power by the SNR. This step determines the relative 'loudness' of the noise compared to the signal.
### Generate Synthetic Environmental Noise
- Environmental noise is simulated using a normal (Gaussian) distribution (`np.random.normal`) with a mean of 0 and a standard deviation derived from the calculated noise power.
- The shape of the noise array matches the input signal, allowing for a direct addition to the signal.
### Add Noise to the Signal
The generated noise is added to the original signal, creating a noisy version of the signal.
### Clip the Noisy Signal
The noisy signal is clipped to ensure that its values remain within the range of -1.0 to 1.0. This step is crucial for preventing distortion, especially in audio processing.

## 2. Artifact noise
Here the noise is simulated by generation random noise with structured, periodic sporadic disturbances that mimic real-life artifacts such as equipment movements or sudden, brief interferences. Guassian distribution is used to generate random then on top of that regular spikes is superimposed, idea is to mimic real word artifact noise by combining random noise with the characteristic suddenness of spikes.
### Calculate the Power of the Signal
The power of the signal is computed as the mean of the squared signal values, which is a standard method to estimate the energy of the signal.
### Determine Required Noise Power for Desired SNR
- SNR (Signal-to-Noise Ratio) is used to quantify the level of the desired signal relative to the level of background noise.
- The desired SNR is given in decibels (dB) and converted to a linear scale using the formula `10 ** (desired_snr_db / 10)`.
- The noise power is calculated by dividing the signal power by this linear SNR value. This process determines the energy level of the noise in relation to the signal.
### Generate Base Artificial Noise
- Artificial noise is initially generated as a Gaussian (normal) distribution with a mean of 0 and a standard deviation derived from the calculated noise power. This forms the base layer of the artifact noise.
### Add Sporadic Intense Noise Spikes
- To simulate artifacts like stethoscope movement or other intermittent disturbances often encountered in medical recordings, the function introduces sporadic intense noise spikes.
- These spikes are added at regular intervals (every 50 samples in the provided example), and the magnitude of each spike is also derived from the Gaussian distribution with the calculated noise power.
### Combine Noise with the Signal
- The artificial noise (base noise plus spikes) is added to the original signal, resulting in a signal that now contains synthesized artifact noise.
### Clip the Noisy Signal
- The resulting noisy signal is clipped to ensure that its values remain within the range of -1.0 to 1.0. This is crucial for maintaining the integrity of the signal and preventing distortion, particularly important in audio processing.
## 3. Respiratory noise
This function mimics the respiratory noise. At first, a time array i created. The "time array" refers to an array where each element represents a specific point in time corresponding to each sample in the audio signal. Using this time array, an sinosudial wave is generated with 0.2 Hz (0.2 is selected to produce 12 breaths per minute 0.2 x 60 sec = 12 breath). The sinusoidal model is a simplified but effective way to mimic the periodic nature of breathing.

### 1. Calculate Signal Power
- The power of the input signal is computed as the average of the squared values of the signal (`np.mean(signal ** 2)`).
### 2. Calculate Required Noise Power for Desired SNR
- SNR is converted from decibels to a linear scale (`10 ** (desired_snr_db / 10)`).
- The noise power needed to achieve this SNR is then determined by dividing the signal power by the linear SNR.
### 3. Create Time Array
- A time array corresponding to each sample in the signal is generated (`np.arange(len(signal)) / sampling_rate`).
### 4. Generate Respiratory Noise
- Respiratory noise is simulated as a sinusoidal wave at a frequency of 0.2 Hz (`np.sin(2 * np.pi * 0.2 * time)`), mimicking a normal adult respiratory rate at rest.
### 5. Scale Respiratory Noise
- The respiratory noise is scaled to match the calculated noise power. This normalization ensures that the noise has the same power as the required noise power.
### 6. Add Respiratory Noise to Signal
- The scaled respiratory noise is added to the original signal, resulting in a noisy signal.
### 7. Clip Noisy Signal
- The noisy signal is clipped to ensure that its values remain within the -1.0 to 1.0 range (`np.clip(noisy_signal, -1.0, 1.0)`). This step is crucial to prevent signal clipping or distortion.
### 8. Return Noisy Signal
- The function returns the signal with the added respiratory noise.

