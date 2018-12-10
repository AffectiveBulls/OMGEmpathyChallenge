

import numpy as np 
import pandas as pd
import os
import cv2

training_img = '/home/AfftectiveBulls/OMG/testset'

subject_name = 'Subject_'



def merge_img(TorV):
    if TorV == 'Test':
        img_data = training_img
        annoit_data = training_annoit
        result_dir = "/home/AfftectiveBulls/OMG/testMerged"
    
    if not os.path.exists(result_dir):
    	os.mkdir(result_dir)
    filenames = []
    f = open('../Frames_Count.txt','r')
    for line in f:
        x = line.split()
        if x:
            filenames.append(x)

    for d in filenames:
    	for index in range(0,int(d[-2])):
    		actimg_path = os.path.join(img_data,os.path.splitext(d[0])[0]+'.mp4','Actor',str(index)+'.png')
    		subimg_path = os.path.join(img_data,os.path.splitext(d[0])[0]+'.mp4','Subject',str(index)+'.png')
    		if os.path.exists(actimg_path) and os.path.exists(subimg_path):
    			actimg = cv2.imread(actimg_path)
    			subimg = cv2.imread(subimg_path)
    			actimg = actimg[61:189, 61:189].astype(np.float32)
    			subimg = subimg[61:189, 61:189].astype(np.float32)
    			actimg = cv2.resize(actimg, (100, 100))
    			subimg = cv2.resize(subimg, (100, 100))


    			saveimg_path = os.path.join(result_dir,os.path.splitext(d[0])[0]+'.mp4')
    			if not os.path.exists(saveimg_path):
    				os.mkdir(saveimg_path)
    			saveimg_path = os.path.join(saveimg_path, str(index)+'.png')
    			saveimg = np.concatenate((actimg, subimg),axis=1)
    			cv2.imwrite(saveimg_path, saveimg)


merge_img('Test')
