## Datasets available

1. 2016 PhysioNet/CinC HEART SOUND - PHS(Processed)
2. International Conference on Biomedical Health Informatics- ICBHI(Processed)
3. OPEN-ACCESS HEART SOUND - OAHS Dataset
4. Hospital Ambient Noise (HAN) Dataset
5. PaHS_Dataset

6. The CirCor DigiScope Phonocardiogram dataset
7. Dog heart beat dataset 


### First model trained on :
1. PHS
2. ICBHI


### Second model trained on :
1. PHS
2. ICBHI
3. HAN
4. PaHS
5. The CirCor DigiScope Phonocardiogram


### Two noise-free datasets :
1. OAHS
2. Dog heart beat (not all but selected samples)



## Contents
1. Introduction
2. Background (about denoising in different applications)
3. State of the art

   ### RQ-2 What are the state-of-the-art audio denoising methods?
   
4. Methodology
   
   4.1 Workflow diagram
   
   4.2 LUnet method and Training
   
   4.3 Metrics
5. Datasets
6. Noise present in Heartbeat sound

   ### RQ-1 What are the various types of noise found in heartbeat audio files, and what methods can be employed to introduce artificial noise into heartbeat audio signals?
   
   6.1 Different noises present in heartbeat audio
   
   6.2 Artificial noise creation and addition to the audio
7. Experimentation and Evaluation

   ### RQ-3 To what extent do denoising methods enhance the fidelity of human heartbeat data, and how does the efficacy of deep learning technique compare to traditional denoising methods in this context?
   
     7.1 Denoising on Human dataset
   
     7.2 Tranfer to Dog heartbeat dataset
    ### RQ-4 How can this deep learning denoising method and model be transferred to the denoising of dog heartbeat data?
   
     7.3 Comparison with traditional denoising method - Variational mode decomposition (VMD)
   
     7.4 Evaluation summary
8. Conclusion 
