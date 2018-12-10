# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:17:06 2018

@author: ntweat
"""

import numpy as np
import pandas as pd
import os
import cv2
#from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
from keras.models import Model
from shutil import copy2


import sys
#####Changable Parameters#######
training_img = '/home/AfftectiveBulls/OMG/testMerged'

ubject_name = 'Subject_'

image_size = 100

model_location = 'savethis24.h5'

sub_numb = ['1','2','3','4','5','6','7','8','9','10']


model_dir = 'ModeMerge'

deepfeat_dir = 'DeepFeatMerge'

batch_size = 50

##########################


def load_video(num_frames, img_data, dirname):
    X_data = []
    for index in range(0, int(num_frames)):
        img_path = os.path.join(img_data,os.path.splitext(dirname)[0]+'.mp4',str(index)+'.png')
        print(img_path)
        if not os.path.exists(img_path):
            img_path = os.path.join(img_data,os.path.splitext(dirname)[0]+'.mp4',str(index+1)+'.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_data,os.path.splitext(dirname)[0]+'.mp4',str(index-1)+'.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_data,os.path.splitext(dirname)[0]+'.mp4',str(index+2)+'.png')
        if True:
            img = cv2.imread(img_path)
            img = img.transpose((2,0,1))
            X_data.append(img)

    return X_data
    

model = load_model(model_location)
extact = Model(inputs=model.input,outputs=model.get_layer('dense_1').output)


def check_sn(num):
    if True:
        img_data = training_img
        result_dir = model_dir
        deep_dir = os.path.join(deepfeat_dir, 'Test')

    subject_name = ubject_name + num + '_'

    filenames = []
    f = open('../Frames_Count.txt','r')
    for line in f:
        x = line.split()
        if x:
            filenames.append(x)


    for d in filenames:
        if (d[0].startswith(subject_name)):
            print(d[0])
            X_data = load_video(d[-2], img_data, d[0])
            X_data = np.asarray(X_data, dtype=np.float32)
            
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            result_file = os.path.join(result_dir,d[0])
            deep_file = os.path.join(deep_dir,d[0])
            Result_data = model.predict(X_data, batch_size=batch_size, verbose=1)
            Deep_data = extact.predict(X_data)

            compare = pd.DataFrame(np.negative(Result_data), columns=['valence'])
            compare.to_csv(result_file)
            cff = pd.DataFrame(Deep_data)
            cff.to_csv(deep_file)
            
            
for sn in sub_numb:
    check_sn(sn)
        
