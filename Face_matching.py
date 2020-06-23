# -*- coding: utf-8 -*-

import face_recognition
import cv2
import numpy as np
import pickle
import operator
import os

directory_pth = 'F:/Desktop/INterestship/Session-8/'


"""
1. read the pickle file. ---> dictionary={embeddings:[]
                                          names:[]}
2. list_embs = dictionary[embeddings] --> 1500 emds (100 celebs)
   list_names = dictionary[names]
   
3. pass an input image (query image)
4. read this image
5. face detection on image
6. feature extrac (VGG) model.predict() ---> test_emd
7. matching:
    1. euclidean distance 
    2. cosine similarity 
    3. SVM classifier (100 clusters)
    4. KNN classifier (100 clusters)

"""


with open(os.path.join(directory_pth,'celeb_embedding.pickle'), 'rb') as f:
    data = pickle.load(f)

f.close()

#print(data.items())
known_face_encodings = data["embeddings"]
known_face_imgpath = data["paths"]
known_face_names = data["names"]

frame = cv2.imread(os.path.join(directory_pth,'Test_Images/SJ2.jpg'))
rgb_frame = frame[:, :, ::-1]

# Find all the faces and face enqcodings in the frame of video
face_locations = face_recognition.face_locations(rgb_frame)
face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

# Loop through each face in this frame of video
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.6)
    
    name = "Unknown"
    counts = {}
    maxval = 0
 
    #face_distance function gives us the distance of test face encoding will all saved encodings
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)  #least distance match
    
    
    if matches[best_match_index]:
        print(best_match_index)
        best_name = known_face_names[best_match_index]
        img_path = known_face_imgpath[best_match_index]
    print("Least distance name : ",best_name)


#displaying the test image and the best match image
Celeb_imagePth = os.path.join(directory_pth, img_path) 
print(Celeb_imagePth)   

celeb_img = cv2.imread(Celeb_imagePth)
celeb_img = cv2.resize(celeb_img, (360, 420))
test_img = cv2.resize(frame, (360, 420))

img_concate_Hori=np.concatenate((test_img,celeb_img),axis=1)

cv2.imshow("Output", img_concate_Hori)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
    