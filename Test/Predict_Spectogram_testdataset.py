# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 20:13:54 2018

@author: Md Taufeeq  Uddin, SK Rahatul Jannat, Ashta Sharma
"""

import os
import numpy as np

from scipy import signal
from scipy.io import wavfile
from sklearn.externals import joblib

numpy_path_ = 'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Test_spectogram/Numpy/'
csv_path_ =  'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Test_spectogram/CSV/'

def load_sound_files(file_paths):
   raw_sounds = []
   file_paths = os.listdir(file_paths)
   for fp in file_paths:
       raw_sounds.append(fp)
   return raw_sounds


#Get_Spectrogram (number 0f features= 366)
def get_spectogram(f, fname):
   
   
   sample_rate, samples = wavfile.read(f)

   
   frequencies, times, spectrogram = signal.spectrogram(samples,sample_rate,nperseg=731)
   number_0f_features = frequencies
   
   #print(spectrogram.shape)
   spectrogram = np.transpose(spectrogram)
   
   base= fname
   new = os.path.splitext(base)[0]
   new1 = os.path.splitext(new)[0]
   np.save(numpy_path_+new1+'.npy', spectrogram)
   #print(new)
   if( new == "Subject_3_Story_3.mp4"):
      dummy = [0] * number_0f_features
      spectrogram = np.vstack([spectrogram, dummy])
   
   np.savetxt(csv_path_+new1+'.csv', spectrogram)
    
   
   return spectrogram

#stack all data
def d_stack(path_):
   sound_files = load_sound_files(path_)
   print(sound_files)
  
   L = []
   for fn in range(len(sound_files)):
       spectogram_features = get_spectogram(path_ + sound_files[fn], fname=sound_files[fn])
       
       for r in spectogram_features:
           L.append(r.tolist())
       
   

   return np.array(L)


#process audio data
path_Xt = 'C:/Users/Raha/Research_data/OMG_empathy/audio/audio/'

X_test = d_stack(path_Xt)


#save processed Data 
resultdir = 'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Result/'
 
np.save(resultdir + '/Test_spectogram.npy', X_test)
print(X_test.shape)
np.savetxt(resultdir + '/Test_spectogram.csv', X_test, delimiter=",")


#Generate prediction
saved_path_ = 'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Result/'
X_test_spectogram_features = np.load(saved_path_ + "Test_spectogram.npy")

path_ = 'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Result/'
model_name= path_ + "omg_emp_model_rf_spectogram.joblib.pkl"
omg_emp_model_rf_land_sub = joblib.load(model_name)


prediction_test_spectogram_features = omg_emp_model_rf_land_sub.predict(X_test_spectogram_features)
X = prediction_test_spectogram_features.reshape(prediction_test_spectogram_features.shape[0],1)
#print(prediction_test_spectogram_features)
#print(X)
np.save(resultdir + '/Prediction_test_deepfeatures.npy', X)
Xcsv = np.load(resultdir + '/Prediction_test_deepfeatures.npy')

np.savetxt(resultdir + '/Prediction_est_spectogram.csv', prediction_test_spectogram_features, delimiter=",")

