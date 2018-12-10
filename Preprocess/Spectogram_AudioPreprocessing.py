# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 20:13:54 2018

@author: Md Taufeeq  Uddin, SK Rahatul Jannat, Ashta Sharma
"""

import os

import numpy as np
import pandas as pd

#Spectrogram
from scipy import signal
from scipy.io import wavfile



def load_sound_files(file_paths):
   raw_sounds = []
   file_paths = os.listdir(file_paths)
   for fp in file_paths:
       raw_sounds.append(fp)
   return raw_sounds

def get_spectogram(f, fname):
   
   
   sample_rate, samples = wavfile.read(f)
   
   frequencies, times, spectrogram = signal.spectrogram(samples,sample_rate,nperseg=731)
  
   
   #print(spectrogram.shape)
   spectrogram = np.transpose(spectrogram)
 
   
   return spectrogram

def d_stack(path_):
   sound_files = load_sound_files(path_)
   print(sound_files)
   
   L = []
   for fn in range(len(sound_files)):
       spectogram_features = get_spectogram(path_ + sound_files[fn], fname=sound_files[fn])
       
       for r in spectogram_features:
           L.append(r.tolist())
       
   

   return np.array(L)

# vertically stack all valence scores of training set, validation set and test set, separately. 
def y_stack(path_):
    list_of_files = load_sound_files(path_)
    print(list_of_files)

   
    L = []
    for fn in range(len(list_of_files)):
        
        file= pd.read_csv(path_ + str(list_of_files[fn]), engine='python')
        X = file.as_matrix()
       
        for r in X:
           
            L.append(r.tolist())
            
    return np.array(L)


#trainning Audio processing
path_ = 'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Training/' #change path accordigly

resultdir = 'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Result/' #change path accordigly


X_train = d_stack(path_)
np.save(resultdir + '/Train_spectogram.npy', X_train)
np.savetxt(resultdir + '/Train_spectogram.csv', X_train, delimiter=",")


# Validation  Audio processing

path_Xt = 'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Validation/'  #change path accordigly

X_test = d_stack(path_Xt) 
np.save(resultdir + '/Val_spectogram.npy', X_test)
np.savetxt(resultdir + '/Val_spectogram.csv', X_test, delimiter=",")



#get numpy array for  training annotation data
path_yt = 'C:/Users/Raha/Research_data/OMG_empathy/OMG_Empathy2019/Training/Annotations/' #change path accordigly


y_train = y_stack(path_yt)
np.save(resultdir + '/Train_annotation.npy', y_train)
np.savetxt(resultdir + '/Train_annotation.csv', y_train, delimiter=",")


#get numpy array for  validation annotation data
path_yt = 'C:/Users/Raha/Research_data/OMG_empathy/OMG_Empathy2019/Validation/Annotations/'
Y_test = y_stack(path_yt)
np.save(resultdir + '/Val_annotation.npy', Y_test)
np.savetxt(resultdir + '/Val_annotation.csv', Y_test, delimiter=",")

