
import numpy as np 
import pandas as pd
import os
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K
K.set_image_dim_ordering('th')

import sys

training_img = 'Training'
validation_img = 'Validation'
training_annoit = 'Training/Annotations'
validation_annoit = 'Validation/Annotations'
subject_name = 'Subject_'

subjectoractor = 'Subject'

image_size = 100

def create_model():
    nb_filters = 8
    nb_conv = 5

    model = Sequential()
    model.add(Convolution2D(8, (5, 5),
                            border_mode='valid',
                            input_shape=(3,image_size, image_size),data_format='channels_first' ) )
    model.add(Activation('relu'))

    model.add(Convolution2D(8, 5, 5))
    model.add(Activation('relu'))

    model.add(Convolution2D(8, 5, 5))
    model.add(Activation('relu'))

    model.add(Convolution2D(8, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(16, 5,5))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 5, 5))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 5, 5))
    model.add(Activation('relu'))

    model.add(Convolution2D(16,5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model

def load_data(TorV):
    print(TorV)
    if TorV == 'Training':
        img_data = training_img
        annoit_data = training_annoit
    else:
        img_data = validation_img
        annoit_data = validation_annoit
    
    directories = os.listdir(annoit_data)
    X_data = []
    Y_data = []
    for dirname in directories:
        if (dirname.startswith(subject_name)):
            print(dirname)
            csv_loaded = pd.read_csv(os.path.join(annoit_data,dirname))
            for index, row in csv_loaded.iterrows():
                img_path = os.path.join(img_data,os.path.splitext(dirname)[0]+'.mp4',subjectoractor,str(index)+'.png')
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img = img[75:175, 75:175].astype(np.float32)
                    img = cv2.resize(img, (image_size, image_size))

                    img = img.transpose((2,0,1))
                    X_data.append(img)
                    Y_data.append(csv_loaded['valence'][index])
                    
    return X_data, Y_data
            



def train_model(batch_size = 50, nb_epoch = 20):
    model = create_model()

    X_train, Y_train = load_data('Training')
    X_Val, Y_Val = load_data('Validation')
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    X_Val = np.asarray(X_Val, dtype=np.float32)
    Y_Val = np.asarray(Y_Val, dtype=np.float32)

    checkpoint = ModelCheckpoint("Best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_Val, Y_Val), callbacks = [checkpoint, early])
    predictions_valid = model.predict(X_Val, batch_size=50, verbose=1)
    model.save('Final_Model.h5')
    return model

train_model(nb_epoch=10000)
