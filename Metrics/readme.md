# Here metrics used to evaluate the results is discussed

## SNR

The `estimate_snr` function calculates the Signal-to-Noise Ratio (SNR) between an original signal and its enhanced version. SNR is a key metric in evaluating signal quality, especially in signal processing and audio enhancement fields. Below is a detailed breakdown of the function's operations:

### 1. Aligning Signal Lengths
- The lengths of the original and enhanced signals are equalized to ensure accurate SNR calculation.
- `min_length = min(len(original), len(enhanced))` determines the length of the shorter signal.
- Both signals are trimmed to this minimum length for consistency.
### 2. Calculating Noise
- Noise is computed as the difference between the original and enhanced signals (`noise = original - enhanced`).
### 3. Calculating Noise Power
- The noise power is estimated using `np.mean(noise**2)`, which is the average of the squared values of the noise.
- A conditional check (`if np.mean(noise**2) > 0 else 1e-7`) is used to avoid zero power values, which would be problematic in later calculations.
### 4. Calculating Signal Power
- The power of the enhanced signal is calculated similarly using `np.mean(enhanced**2)`.
- The same zero-value check is applied to ensure a non-zero power value.
### 5. Calculating SNR
- SNR is calculated using `10 * np.log10(signal_power / noise_power)`, converting the power ratio to a logarithmic scale in decibels (dB).
- A higher SNR indicates a better quality signal where the desired signal is more prominent compared to the noise.
### 6. Return SNR
- The function returns the SNR value, providing a quantitative measure of signal enhancement quality.
