import os
import soundfile as sf
import numpy as np
from tensorflow import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import math
import neptune
from audio_data_augmentation import add_environmental_noise, add_artifact_noise, add_respiratory_noise, add_electronic_interference, add_cable_movement_noise, all_5_noises
from neptune.integrations.tensorflow_keras import NeptuneCallback



def segment_audio(audio, segment_samples, sampling_rate, overlap=0.5):
    hop_samples = int(segment_samples * (1 - overlap))  # Calculate hop size based on overlap
    num_segments = int(np.ceil((len(audio) - segment_samples) / hop_samples)) + 1
    segments = []
    for i in range(num_segments):
        start = i * hop_samples  # Start index for each segment taking into account the overlap
        end = start + segment_samples
        segment = audio[start:end]
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)), 'constant')
        segments.append(segment)
    return np.array(segments)

def combine_segments(segments, sampling_rate, overlap=0.5):
    segment_samples = segments.shape[1]
    hop_samples = int(segment_samples * (1 - overlap))  # Calculate hop size based on overlap
    audio_length = hop_samples * (segments.shape[0] - 1) + segment_samples
    audio = np.zeros(audio_length)
    for i, segment in enumerate(segments):
        start = i * hop_samples  # Start index for adding segments back together with overlap
        end = start + segment_samples
        audio[start:end] += segment[:end-start]  # Add the segment with overlap
    return audio

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

# def SI_SNR(est, egs):
#     '''
#          Calculate the SNR indicator between the two audios. 
#          The larger the value, the better the separation.
#          input:
#                _s: Generated audio
#                s:  Ground Truth audio
#          output:
#                SNR value 
#     '''
#     min_length = min(len(est), len(egs))
#     est = est[:min_length]
#     egs = egs[:min_length]
#     # Calculate noise
#     noise = est - egs
#     noise_power = np.sum(np.power(noise, 2)) / len(noise)
#     signal_power = np.sum(np.power(egs, 2)) / len(egs)
#     snr = 10 * math.log10(signal_power / noise_power)
#     return snr

def calculate_mse(original_signal, processed_signal):
    # Ensure both signals are of the same length
    # if len(original_signal) != len(processed_signal):
    #     raise ValueError("The length of both signals must be the same")

    min_length = min(len(original_signal), len(processed_signal))
    original_signal = original_signal[:min_length]
    processed_signal = processed_signal[:min_length]
    # Calculate MSE
    mse = np.mean((original_signal - processed_signal) ** 2)
    return mse


# Use the pretrained weights from Models folder
# model = keras.models.load_model('/work/pbenga2s/Pytorch/thesis/lunet/lunet_model_pc07_PhysioNet_1000_-3dB_to_6dB_old.h5') 
# model = keras.models.load_model("/work/pbenga2s/Pytorch/thesis/lunet/lunet_model_pc07_PhysioNet_1000_-3dB_to_6dB_(trained_on_5_datasets).h5")
model = keras.models.load_model('/work/pbenga2s/Pytorch/thesis/lunet/saved_models/2data/lunet_model_pc07_PhysioNet_1000_-3dB_to_6dB_(2data).h5') 


# input_directory = '/work/pbenga2s/Pytorch/thesis/lunet/sound_data/'  
# output_directory = '/work/pbenga2s/Pytorch/thesis/lunet/sound_data_output/'
# spectrogram_directory = '/work/pbenga2s/Pytorch/thesis/lunet/sound_data_output/specto'
# snr_mse_results_file = os.path.join(output_directory, '/work/pbenga2s/Pytorch/thesis/lunet/sound_data_output/snr_results.txt')


# input_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Data/OAHS_Dataset/'  
# output_directory = '/work/pbenga2s/Pytorch/thesis/lunet/OAHS_data_output(5snr)/'
# spectrogram_directory = '/work/pbenga2s/Pytorch/thesis/lunet/OAHS_data_output(5snr)/specto'
# snr_results_file = os.path.join(output_directory, '/work/pbenga2s/Pytorch/thesis/lunet/OAHS_data_output(5snr)/snr_results.txt')

# input_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Data/noise_data/data/Files'  
# output_directory = '/work/pbenga2s/Pytorch/thesis/lunet/dog_data_output/dog_noise_data(5snr)/'
# spectrogram_directory = '/work/pbenga2s/Pytorch/thesis/lunet/dog_data_output/dog_noise_data(5snr)/specto'
# snr_results_file = os.path.join(output_directory, '/work/pbenga2s/Pytorch/thesis/lunet/dog_data_output/dog_noise_data(5snr)/snr_results.txt')

# input_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Data/noise_data/data/Files'  
# output_directory = '/work/pbenga2s/Pytorch/thesis/lunet/dog_data_output/dog_noise_data(3snr)/'
# spectrogram_directory = '/work/pbenga2s/Pytorch/thesis/lunet/dog_data_output/dog_noise_data(3snr)/specto'
# snr_results_file = os.path.join(output_directory, '/work/pbenga2s/Pytorch/thesis/lunet/dog_data_output/dog_noise_data(3snr)/snr_results.txt')


# # code for model trained with 5 datasets for dog data
# input_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Data/noise_data/data/Files'  
# output_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_5datatrained/dog_data/dog_noise_data(2snr)/'
# spectrogram_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_5datatrained/dog_data/dog_noise_data(2snr)/specto'
# snr_results_file = os.path.join(output_directory, '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_5datatrained/dog_data/dog_noise_data(2snr)/snr_results.txt')


# # code for model trained with 2 datasets for human data
# input_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Data/oahs(all_in_one)'    
# output_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_2datatrained/human_data/human_noise_data(-5snr)/'
# spectrogram_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_2datatrained/human_data/human_noise_data(-5snr)/specto'
# snr_mse_results_file = os.path.join(output_directory, '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_2datatrained/human_data/human_noise_data(-5snr)/snr_results.txt')
desired_snr_db = 0  # Adjust noise levels as needed

# input_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Data/noise_data/data/Files'  
# output_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_2datatrained/dog_data/dog_noise_data(0snr)/'
# spectrogram_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_2datatrained/dog_data/dog_noise_data(0snr)/specto'
# snr_mse_results_file = os.path.join(output_directory, '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_2datatrained/dog_data/dog_noise_data(0snr)/snr_results.txt')

# input_directory='/scratch/pbenga2s/Pytorch/thesis/lunet/Data/Pascal_test'
# input_directory='/scratch/pbenga2s/Pytorch/thesis/lunet/Data/noise_data/data/Files'

# output_directory = '/scratch/pbenga2s/Pytorch/thesis/lunet/dog/'  # Folder to save processed audio files
# spectrogram_directory = '/scratch/pbenga2s/Pytorch/thesis/lunet/dog/specto'
# snr_results_file = os.path.join(output_directory, '/scratch/pbenga2s/Pytorch/thesis/lunet/dog/snr_results.txt')

run = neptune.init_run(
        project='prabhudev/lunet',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYzRjZTQ3OS01OWQwLTRhNjktYmFmMi1mMDgxM2U3NTg2YTYifQ==',
    )

# snr_mse_table = Table(columns=['Filename', 'Noise Added SNR', 'Model Processed SNR', 'Alternative NR SNR', 'Noise Added MSE', 'Model Processed MSE', 'Alternative NR MSE'])



# # Create the output directory if it doesn't exist
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# Model's expected input length for each segment in samples
expected_input_length = 800  # Number of samples that the model expects for each segment

def save_mel_spectrogram(signal, sr, file_path,logger):
    S = librosa.feature.melspectrogram(y=signal, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(file_path)
    # logger['images/'].upload(file_path)
    logger.upload(file_path)
 
    plt.close()

def save_waveform(signal, filename, title, sampling_rate):
    time = np.linspace(0, len(signal) / sampling_rate, num=len(signal))
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.savefig(filename)
    plt.close()

# Specify noise levels for different types of noise

# artifact_noise_level = 500
# respiratory_noise_level = 800
neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

# with open(snr_mse_results_file, 'w') as file:
#     # Write the header to the file
#     file.write("Filename, Noise Added SNR (r-n), Model Processed SNR (n-d)(dB), Noise Reduction SNR (r-d)(dB), Noise Added MSE (r-n), Model Processed MSE (n-d), Alternative Noise Reduction (r-d)\n")

#     # Process files in the input directory
#     for filename in os.listdir(input_directory):
#         if filename.endswith('.wav'):  # Check if the file is a WAV file
#             # Load the audio file
#             file_path = os.path.join(input_directory, filename)
#             signal, sr = sf.read(file_path)
        

#             original_spectrogram_path = os.path.join(spectrogram_directory, f'{filename}_original.png')
#             # save_mel_spectrogram(signal, sr, original_spectrogram_path,logger=run)['images/']
#             save_mel_spectrogram(signal, sr, original_spectrogram_path,logger=run['images/'+filename+'/original/'])
            
#             # run.upload(f'metrics/original_spectrogram/{filename}', signal)
#             # run.upload('metrics/noisy_spectrogram', original_spectrogram_path)
#             # run.upload('metrics/noisy_waveform', noisy_waveform_path)
#             # run.upload('metrics/processed_spectrogram', processed_spectrogram_path)
#             # run.upload('metrics/pynr_processed_spectrogram', pynr_processed_spectrogram_path)


#             # Save the waveform of the original audio
#             original_waveform_path = os.path.join(spectrogram_directory, f'{filename}_original_waveform.png')
#             save_waveform(signal, original_waveform_path, 'Original Waveform', sr)
#             # neptune.log_image(f'metrics/original_waveform/{filename}', signal)

#             org_signal = signal
#             # Combine all types of noise
#             noisy_pcg_all_noises = all_5_noises(signal, desired_snr_db, 50, 0.3, 0.3, sr)

#             noise_added_snr = estimate_snr(signal, noisy_pcg_all_noises)
#             noise_added_mse = calculate_mse(signal, noisy_pcg_all_noises)
#             signal = noisy_pcg_all_noises
            
#             original_spectrogram_path = os.path.join(spectrogram_directory, f'{filename}_noise_added.png')
#             # save_mel_spectrogram(signal, sr, original_spectrogram_path,logger=run)
#             save_mel_spectrogram(signal, sr, original_spectrogram_path,logger=run['images/'+filename+'/noise_added/'])
#             # neptune.log_image(f'metrics/noisy_spectrogram/{filename}', signal)

#             # Save the waveform of the audio after adding noise
#             noisy_waveform_path = os.path.join(spectrogram_directory, f'{filename}_noisy_waveform.png')
#             save_waveform(signal, noisy_waveform_path, 'Noisy Waveform', sr)
#             # neptune.log_image(f'metrics/noisy_waveform/{filename}', signal)

#             # Save the combined audio to the output directory
#             output_file_path = os.path.join(output_directory, ('noise_added_'+filename))
#             sf.write(output_file_path, signal, sr)
#             print(f"Processed and saved noise audio: {output_file_path}")

#             # Segment the audio
#             segments = segment_audio(signal, expected_input_length, sr, overlap=0.5)

#             # Process each segment with the model
#             processed_segments = np.array([model.predict(s[np.newaxis, ..., np.newaxis]) for s in segments])

#             # Remove extra dimensions added by the model
#             processed_segments = np.squeeze(processed_segments)

#             # Combine segments back into a single audio signal
#             combined_signal = combine_segments(processed_segments, sr, overlap=0.5)

#             # Create and save the mel spectrogram for the denoised audio
#             processed_spectrogram_path = os.path.join(spectrogram_directory, f'{filename}_processed.png')
#             # save_mel_spectrogram(combined_signal, sr, processed_spectrogram_path,logger=run)
#             save_mel_spectrogram(combined_signal, sr, processed_spectrogram_path,logger=run['images/'+filename+'/model_processed/'])
#             # neptune.log_image(f'metrics/denoised_spectrogram/{filename}', combined_signal)

#             pynr_method = nr.reduce_noise(y=signal, sr=sr)
#             # pynr_method = nr.reduce_noise(y=signal, sr=sr, time_mask_smooth_ms=256)
#             pynr_processed_spectrogram_path = os.path.join(spectrogram_directory, f'{filename}_pynr_processed.png')
#             # save_mel_spectrogram(pynr_method, sr, pynr_processed_spectrogram_path,logger=run)
#             save_mel_spectrogram(pynr_method, sr, pynr_processed_spectrogram_path,logger=run['images/'+filename+'/pynr_processed/'])
#             # neptune.log_image(f'metrics/python_library_NoiseReduce_denoised_spectrogram/{filename}', pynr_method)



#             # Save the combined audio to the output directory
#             output_file_path = os.path.join(output_directory, filename)
#             sf.write(output_file_path, combined_signal, sr)
#             print(f"Processed and saved: {output_file_path}")

#             # Compute SNR values
#             # original_snr = estimate_snr(clean_signal, original_signal)
#             model_processed_snr = estimate_snr(signal, combined_signal)
#             pynr_processed_snr = estimate_snr(signal, pynr_method)

#             org1 = estimate_snr(org_signal, combined_signal)
#             org2 = calculate_mse(org_signal, combined_signal)

#             model_processed_mse = calculate_mse(signal, combined_signal)
#             pynr_processed_mse = calculate_mse(signal, pynr_method)

#             # Inside your loop, after calculating SNR and MSE
#             # snr_mse_table.add_data(filename, noise_added_snr, model_processed_snr, pynr_processed_snr, noise_added_mse, model_processed_mse, pynr_processed_mse)
            

#             # model_processed_snr = SI_SNR(combined_signal,signal)
#             # pynr_processed_snr = SI_SNR(pynr_method,signal)

#             # Log the SNR values to Neptune
#             # neptune.log_metric(f'metrics/SNR_noise_added/{filename}_noise_added_snr', noise_added_snr)
#             # neptune.log_metric(f'metrics/SNR_lunet_processed/{filename}_model_processed_snr', model_processed_snr)
#             # neptune.log_metric(f'metrics/SNR_NoiseReduce_library_processed/{filename}_pynr_processed_snr', pynr_processed_snr)


#             # Write SNR results to file
#             file.write(f"{filename}, {noise_added_snr}, {model_processed_snr}, {org1}, {noise_added_mse}, {model_processed_mse}, {org2}\n")

#             print(f"Processed and saved SNR for: {filename}")

# # run['SNR_MSE_Results'] = snr_mse_table
# run["test/metrics"].upload(snr_mse_results_file)





























# without all_5_noises

input_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Data/noise_data/data/Files'  
output_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_2datatrained/dog_data/without_noise/'
spectrogram_directory = '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_2datatrained/dog_data/without_noise/specto'
snr_mse_results_file = os.path.join(output_directory, '/work/pbenga2s/Pytorch/thesis/lunet/Output_of_2datatrained/dog_data/without_noise/snr_results.txt')

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


with open(snr_mse_results_file, 'w') as file:
    # Write the header to the file
    file.write("Filename, SNR, MSE\n")

    # Process files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.wav'):  # Check if the file is a WAV file
            # Load the audio file
            file_path = os.path.join(input_directory, filename)
            signal, sr = sf.read(file_path)
        

            original_spectrogram_path = os.path.join(spectrogram_directory, f'{filename}_original.png')
            # save_mel_spectrogram(signal, sr, original_spectrogram_path,logger=run)['images/']
            save_mel_spectrogram(signal, sr, original_spectrogram_path,logger=run['images/'+filename+'/original/'])

            # Save the waveform of the original audio
            original_waveform_path = os.path.join(spectrogram_directory, f'{filename}_original_waveform.png')
            save_waveform(signal, original_waveform_path, 'Original Waveform', sr)
            # neptune.log_image(f'metrics/original_waveform/{filename}', signal)

            org_signal = signal
            # Combine all types of noise
            # noisy_pcg_all_noises = all_5_noises(signal, desired_snr_db, 50, 0.3, 0.3, sr)

            # noise_added_snr = estimate_snr(signal, noisy_pcg_all_noises)
            # noise_added_mse = calculate_mse(signal, noisy_pcg_all_noises)
            # signal = noisy_pcg_all_noises
            
            # original_spectrogram_path = os.path.join(spectrogram_directory, f'{filename}_noise_added.png')
            # # save_mel_spectrogram(signal, sr, original_spectrogram_path,logger=run)
            # save_mel_spectrogram(signal, sr, original_spectrogram_path,logger=run['images/'+filename+'/noise_added/'])
            # # neptune.log_image(f'metrics/noisy_spectrogram/{filename}', signal)

            # # Save the waveform of the audio after adding noise
            # noisy_waveform_path = os.path.join(spectrogram_directory, f'{filename}_noisy_waveform.png')
            # save_waveform(signal, noisy_waveform_path, 'Noisy Waveform', sr)
            # # neptune.log_image(f'metrics/noisy_waveform/{filename}', signal)

            # Save the combined audio to the output directory
            # output_file_path = os.path.join(output_directory, ('noise_added_'+filename))
            # sf.write(output_file_path, signal, sr)
            # print(f"Processed and saved noise audio: {output_file_path}")

            # Segment the audio
            segments = segment_audio(signal, expected_input_length, sr, overlap=0.5)

            # Process each segment with the model
            processed_segments = np.array([model.predict(s[np.newaxis, ..., np.newaxis]) for s in segments])

            # Remove extra dimensions added by the model
            processed_segments = np.squeeze(processed_segments)

            # Combine segments back into a single audio signal
            combined_signal = combine_segments(processed_segments, sr, overlap=0.5)

            # Create and save the mel spectrogram for the denoised audio
            processed_spectrogram_path = os.path.join(spectrogram_directory, f'{filename}_processed.png')
            # save_mel_spectrogram(combined_signal, sr, processed_spectrogram_path,logger=run)
            save_mel_spectrogram(combined_signal, sr, processed_spectrogram_path,logger=run['images/'+filename+'/model_processed/'])
            # neptune.log_image(f'metrics/denoised_spectrogram/{filename}', combined_signal)

            # pynr_method = nr.reduce_noise(y=signal, sr=sr)
            # # pynr_method = nr.reduce_noise(y=signal, sr=sr, time_mask_smooth_ms=256)
            # pynr_processed_spectrogram_path = os.path.join(spectrogram_directory, f'{filename}_pynr_processed.png')
            # # save_mel_spectrogram(pynr_method, sr, pynr_processed_spectrogram_path,logger=run)
            # save_mel_spectrogram(pynr_method, sr, pynr_processed_spectrogram_path,logger=run['images/'+filename+'/pynr_processed/'])
            # # neptune.log_image(f'metrics/python_library_NoiseReduce_denoised_spectrogram/{filename}', pynr_method)



            # Save the combined audio to the output directory
            output_file_path = os.path.join(output_directory, filename)
            sf.write(output_file_path, combined_signal, sr)
            print(f"Processed and saved: {output_file_path}")

            # Compute SNR values
            # original_snr = estimate_snr(clean_signal, original_signal)
            snr = estimate_snr(signal, combined_signal)
            # pynr_processed_snr = estimate_snr(signal, pynr_method)

            # org1 = estimate_snr(org_signal, combined_signal)
            # org2 = calculate_mse(org_signal, combined_signal)

            mse = calculate_mse(signal, combined_signal)
            # pynr_processed_mse = calculate_mse(signal, pynr_method)

            # Inside your loop, after calculating SNR and MSE
            # snr_mse_table.add_data(filename, noise_added_snr, model_processed_snr, pynr_processed_snr, noise_added_mse, model_processed_mse, pynr_processed_mse)
            

            # model_processed_snr = SI_SNR(combined_signal,signal)
            # pynr_processed_snr = SI_SNR(pynr_method,signal)

            # Log the SNR values to Neptune
            # neptune.log_metric(f'metrics/SNR_noise_added/{filename}_noise_added_snr', noise_added_snr)
            # neptune.log_metric(f'metrics/SNR_lunet_processed/{filename}_model_processed_snr', model_processed_snr)
            # neptune.log_metric(f'metrics/SNR_NoiseReduce_library_processed/{filename}_pynr_processed_snr', pynr_processed_snr)


            # Write SNR results to file
            file.write(f"{filename}, {snr}, {mse}\n")

            print(f"Processed and saved SNR for: {filename}")

# run['SNR_MSE_Results'] = snr_mse_table
run["test/metrics"].upload(snr_mse_results_file)