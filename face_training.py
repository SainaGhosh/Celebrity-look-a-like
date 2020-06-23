# -*- coding: utf-8 -*-
"""
Phase 2: Feature Extarction. 
In this script we detect faces using Dlib (hog) and extract the features using Face Recognition API on the 100 Celeb Dataset. 
The embeddings and its corresponnding name are stored in a pickle file in a dictionary format.
{
     embedings:[emb0, emb1, emb2, emb3, .........],
     names:[name0, name1, name2, name3, .........],
     path: [img1_pth, img2_pth, ..........]   (This can be skipped)
}
The extracted features will then be used while classification.

"""

import cv2
import dlib
import face_recognition
import os
import numpy as np
import pickle

print("Imports done !!!")
folder_path = './dataset'    #100 Celeb dataset. 15 images per celeb

names = os.listdir(folder_path)
total = 0

knownEmbeddings = []
knownNames = []
knownPath = []

detector = dlib.get_frontal_face_detector()  #Dlib Face detector
print("detector loaded !!")


for name in names:    #iterating over the folders in dataset - folders named with celeb names
    print(name)
    img_list = os.listdir(folder_path+'/'+name)
    
    print((img_list))

    cnt = 0
    for img_path in img_list:
        
        pixels = cv2.imread(folder_path+'/'+name+'/'+img_path)
        if pixels is None:
            continue
        if pixels.shape[0]<640 and pixels.shape[1]<480:    #discard face if its size is less than a particular value
            continue
        
        rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        
        if len(boxes)==1: #consider only one face per image. If multiple faces present in the single image, discard them
            cnt += 1
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes,num_jitters=1)

            knownEmbeddings.append(encodings[0])
            knownNames.append(name)
            knownPath.append(folder_path+'/'+name+'/'+img_path)
        if cnt == 15:
            break
    print(name,cnt)
data = {"embeddings": knownEmbeddings, "names": knownNames,"paths":knownPath}

#write the dictionary in the pickle file
f = open('celeb_embedding.pickle', "wb")
f.write(pickle.dumps(data))
f.close()




