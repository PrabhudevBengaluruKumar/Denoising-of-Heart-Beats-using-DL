# path_Heart_Train="/content/drive/MyDrive/Heart Sound Denoising All Data/PHS Data (Processed)/train"
# path_Heart_Train = "/work/pbenga2s/Pytorch/thesis/lunet/Data/PHS(Processed)/train"
path_Heart_Train = "/work/pbenga2s/Pytorch/thesis/lunet/Data/new_train_data/train"

# path_Lung_Train="/content/drive/MyDrive/Heart Sound Denoising All Data/ICBHI Dataset (Processed)/train"
path_Lung_Train = "/work/pbenga2s/Pytorch/thesis/lunet/Data/ICBHI(Processed)/train"

# pathheartVal='/content/drive/MyDrive/Heart Sound Denoising All Data/OAHS Dataset/Git_val'
pathheartVal = "/work/pbenga2s/Pytorch/thesis/lunet/Data/OAHS_Dataset/Git_val"

# pathlungval='/content/drive/MyDrive/Heart Sound Denoising All Data/ICBHI Dataset (Processed)/val'
pathlungval = "/work/pbenga2s/Pytorch/thesis/lunet/Data/ICBHI(Processed)/val"

# pathhospitalval='/content/drive/MyDrive/Heart Sound Denoising All Data/Hospital Ambient Noise (HAN) Dataset'
pathhospitalval = "/work/pbenga2s/Pytorch/thesis/lunet/Data/HAN_Dataset"

pathPascal='/work/pbenga2s/Pytorch/thesis/lunet/Data/PaHS_Dataset'

window_size=.8
name_model="lunet"
input_shape=800
output_shape=800
sampling_rate_new=1000
check=rf"/work/pbenga2s/Pytorch/thesis/lunet/saved_models/5data/{name_model}_model_pc07_PhysioNet_{sampling_rate_new}_-3dB_to_6dB_(5data).h5"
# check=rf"/work/pbenga2s/Pytorch/thesis/lunet/saved_models/2data/{name_model}_model_pc07_PhysioNet_{sampling_rate_new}_-3dB_to_6dB_(2data).h5"
# check=rf"/work/pbenga2s/Pytorch/thesis/lunet/{name_model}_model_pc07_PhysioNet_{sampling_rate_new}_-3dB_to_6dB_(trained_on_5_datasets).h5"
# check=rf"/work/pbenga2s/Pytorch/thesis/lunet/{name_model}_model_pc07_PhysioNet_{sampling_rate_new}_-3dB_to_6dB.h5"
#check=rf"./check_points/{name_model}_model_pc07_PhysioNet_{sampling_rate_new}_-3dB_to_6dB.h5"
