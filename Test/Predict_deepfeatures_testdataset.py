# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:20:57 2018

@author: Md Taufeeq  Uddin, SK Rahatul Jannat, Ashta Sharma
"""

import os
import numpy as np
from sklearn.externals import joblib



numpy_path_ = 'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Test_DeepFeatures/Numpy/'
csv_path_ =  'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Test_DeepFeatures/CSV/'



#Load files From Directory
def load_sound_files(file_paths):
   raw_sounds = []
   file_paths = os.listdir(file_paths)
   for fp in file_paths:
       raw_sounds.append(fp)
   return raw_sounds
 
from numpy import genfromtxt


#get Deep Features (#number of features = 256)
def get_deep_features(path_,fname):
    my_data = genfromtxt(path_, delimiter=',')
    my_data = np.delete(my_data, 0, axis=0)
    my_data = my_data[:,1:]
    base= numpy_path_+fname
    base1= csv_path_+fname
    new = os.path.splitext(base)[0]
    print(new)
    new1= os.path.splitext(base1)[1]
    np.save(new+'.npy', my_data)
    X_train =  np.load(new+'.npy')
    print(X_train.shape)
    np.savetxt(new1+'.csv', X_train)
    
    return my_data


#stack all data
def d_stack(path_):
   sound_files = load_sound_files(path_)
  
   L = []
   for fn in range(len(sound_files)):
       deepfeatures = get_deep_features(path_ + sound_files[fn],fname=sound_files[fn])
       
       for r in deepfeatures:
           L.append(r.tolist())
       
   
   return np.array(L)



#process deep_features data
resultdir = 'C:/Users/Raha/Research_data/OMG_empathy/Training-20181025T230421Z-001/Result'
file_paths = 'C:/Users/Raha/Research_data/OMG_empathy/DeepFeat/DeepFeatMerge/Test/'

X_train = d_stack(file_paths)

#save processed Data 
np.save(resultdir + '/deep_features_test.npy', X_train)

X_train = np.load(resultdir + '/deep_features_test.npy')
print(X_train.shape)
np.savetxt(resultdir + '/deep_features_test.csv', X_train, delimiter=",")


#Generate prediction
X_test_deep_features = np.load(resultdir + "/deep_features_test.npy")
print(X_test_deep_features.shape)

model_name= resultdir + "/omg_emp_model_rf_deep_features.joblib.pkl"
omg_emp_model_rf_land_sub = joblib.load(model_name)


prediction_test_deep_features = omg_emp_model_rf_land_sub.predict(X_test_deep_features)
X = prediction_test_deep_features.reshape(prediction_test_deep_features.shape[0],1)
#print(prediction_test_deep_features)
#print(X)
np.save(resultdir + '/Prediction_test_deepfeatures.npy', X)
Xcsv = np.load(resultdir + '/Prediction_test_deepfeatures.npy')
np.savetxt(resultdir + '/Prediction_test_deepfeatures.csv', Xcsv, delimiter=",")


