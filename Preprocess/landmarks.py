# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 17:06:38 2018

@author: Md Taufeeq  Uddin, SK Rahatul Jannat, Ashta Sharma
"""
import numpy as np 
import pandas as pd
import os
import cv2

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
from imutils import face_utils
import imutils
from sklearn.preprocessing import normalize

import dlib
import sys


image_size = 128

shape_predictor = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
rectp = dlib.rectangle(0, 0, 250, 250)


path_ = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/OMG/faces/Training/Subject_1_Story_2.mp4/Subject/"


# combine landmarks of subjects of training set, validation set and test set, separately. Also for actors.  
def stack_landmarks(path_):
    fld_L, landmarks_all_acts, landmarks_all_subs = actor_subject(path_)
    print(fld_L)

    landmarks_all_acts_L, landmarks_all_subs_L = [], []
    
    for all_acts in range(len(landmarks_all_acts)):
        for each_act in range(len(landmarks_all_acts[all_acts])):
            landmarks_all_acts_L.append(landmarks_all_acts[all_acts][each_act])
        # print(landmarks_all_acts[all_acts], "is Done!")
            
    for all_subs in range(len(landmarks_all_subs)):
        for each_sub in range(len(landmarks_all_acts[all_subs])):
            # landmarks_all_acts_L.append(landmarks_all_acts[all_subs][each_sub])
            landmarks_all_subs_L.append(landmarks_all_subs[all_subs][each_sub])
        # print(landmarks_all_subs[all_subs], "is Done!")

    return np.array(landmarks_all_acts_L), np.array(landmarks_all_subs_L)


# get landmark from actor and subject of each video.
def actor_subject(path_):
    fld_L = sorted(os.listdir(path_))    
    print(fld_L)
    
    landmarks_all_acts, landmarks_all_subs = [], []

    for i in fld_L:
        path_2 = path_ + "/" + i + "/"
        fld_L2 = os.listdir(path_2)
        print(fld_L2)
        
        for j in fld_L2:
            final_path = path_2 + j + "/"
            print(final_path)

            if j == "Actor":
                landmarks_c_images_act = load_image_files(final_path, None)
                landmarks_all_acts.append(landmarks_c_images_act)
            else:
                landmarks_c_images_sub = load_image_files(final_path, None)
                landmarks_all_subs.append(landmarks_c_images_sub)
        # print(fld_L2, "is done!")
                
    return fld_L, landmarks_all_acts, landmarks_all_subs


def load_image_files(path_, rectp):
    landmarks_c_images = [] 
    files = sorted(os.listdir(path_))    
    # print(files)
    
    s_image_names = []
    for i in files:
        img_name =i[:-4]
        s_image_names.append(int(img_name))
        
    s_image_names = sorted(s_image_names)
    # print(s_image_names)    
    
    for fp in s_image_names:
        img = cv2.imread(path_ + "/" + str(fp) + ".png") 
        # print(img.shape)
        # print("FP: ", fp)
        landmarks, rectp = landmark_detector(img, rectp)
        # print(type(landmarks))
        # print("Image: ", fp, "Landmarks: ", landmarks)
        if len(landmarks) == 0:
            landmarks = [0] * 136
            
        landmarks_c_images.append(landmarks)
        
        print(path_, ": Landmarks from Image #", fp, "is appended.")
        
    return landmarks_c_images # img, landmarks_c_images


# get landmark from image
def landmark_detector(img, rectp):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 2)
	if not rects:
		rects = rectp
	if not rects:
		return [], rects
	for (i, d) in enumerate(rects):
		shape = predictor(gray, d)
		if not shape:
			return [], rects
		else:
			shape = face_utils.shape_to_np(shape)
			send_lan = []
			for (x, y) in shape:
				send_lan.append([x,y])
			x_max = max(send_lan, key=lambda item: item[0])[0]
			y_max = max(send_lan, key=lambda item: item[1])[1]
			send_lan = [[x/x_max, y/y_max] for x,y in  send_lan]

	return np.array(send_lan).flatten().tolist(), rectp


if __name__ == "__main__":
    # Training
    training_fld_path = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/OMG/faces/Training/"
    trianing_acts_landmarks, training_subs_landmarks = stack_landmarks(training_fld_path)
    
    save_path = 'C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/'
    np.savetxt(save_path + 'trianing_acts_landmarks.csv', trianing_acts_landmarks, fmt='%.2f', delimiter=',')
    np.savetxt(save_path + 'training_subs_landmarks.csv', training_subs_landmarks, fmt='%.2f', delimiter=',')
    
    
    # Validation 
    validation_fld_path = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/OMG/faces/Validation/"
    validation_acts_landmarks, validation_subs_landmarks = stack_landmarks(validation_fld_path)
    
    save_path = 'C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/'
    np.savetxt(save_path + 'validation_acts_landmarks.csv', validation_acts_landmarks, fmt='%.2f', delimiter=',')
    np.savetxt(save_path + 'validation_subs_landmarks.csv', validation_subs_landmarks, fmt='%.2f', delimiter=',')


    # Test
    test_fld_path = "C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/faces/testset/"
    test_acts_landmarks, test_subs_landmarks = stack_landmarks(test_fld_path)

    save_path = 'C:/Users/Md Taufeeq  Uddin/Projects/OMG Empathy Challenge/'
    np.savetxt(save_path + 'test_acts_landmarks.csv', test_acts_landmarks, fmt='%.2f', delimiter=',')
    np.savetxt(save_path + 'test_subs_landmarks.csv', test_subs_landmarks, fmt='%.2f', delimiter=',')
