import tensorflow
from tensorflow import keras 
from tensorflow.keras.callbacks import ModelCheckpoint
from Codes.model import enhancement_model
from Codes.processing_initial import get_files_and_resample
from Codes.config import *
import os
import neptune
# from neptune.integrations.tensorflow_keras import NeptuneCallback

from neptune.integrations.tensorflow_keras import NeptuneCallback

def main():
    run = neptune.init_run(
        project='prabhudev/lunet',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYzRjZTQ3OS01OWQwLTRhNjktYmFmMi1mMDgxM2U3NTg2YTYifQ==',
    )
    neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

    # model.fit(
    #     x_train,
    #     y_train,
    #     epochs=5,
    #     batch_size=64,
    #     callbacks=[neptune_cbk],
    # )


    # neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

    X1,Y1,labeltrain = get_files_and_resample(sampling_rate_new, window_size, locH=path_Heart_Train, locN=path_Lung_Train, mode=0)
    print(X1.shape)  
    checkpoint = ModelCheckpoint(check, monitor='val_loss', verbose=1, 
                                save_best_only=True, save_weights_only=False, mode='auto')

    model =enhancement_model(name_model=name_model, input_shape=input_shape, output_shape=output_shape, loss_function='mse').model
    history = model.fit(
        X1, Y1, 
        epochs=20, 
        batch_size=128, 
        verbose=1, 
        validation_split=0.1, 
        callbacks=[checkpoint, neptune_cbk]
    )
    # history=model.fit(X1, Y1, epochs=20, batch_size=128, verbose=1, validation_split=0.1, callbacks=[checkpoint])

    run['parameters'] = {'batch_size': 128, 'epochs': 20, 'loss_function': 'mse'}
    run['name_model'] = name_model
    run.stop()

    # X1 = get_files_and_resample_spect(sampling_rate_new, window_size,locH=path_Heart_Train,locN=path_Lung_Train,mode=0)
        
if __name__ == '__main__':
    main()