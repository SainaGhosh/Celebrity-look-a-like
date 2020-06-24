Celebrity Look alike Model FaceNet Keras

This face recognition model comprises of three models:
	1.	Face Detection using MTCNN
	2.	Feature Extraction using FaceNet Keras 
	3.	Feature Classification using SVM Classifier
	
Face Detection using MTCNN:

The FaceDetection_MTCNN.py file (face detection model) uses pretrained MTCNN detector.
This model extracts the face in the image and return the compressed file.
You can try to use a different dataset but the format should be directory>>subdirectory>>images. 
Here, the subdirectory names will be your class label. Add the path to load_dataset(‘write the path here’).
E.g.--> load_dataset(directory/)

Feature Extraction using FaceNet Keras:

The FacenetKeras_Embeddings.py file(feature extraction model) uses the pretrained “facenet_keras.h5” model. 
It extracts the 128 embeddings from the detected face and stores it in a compressed file.
It loads the compressed face data file and generalizes the image to extract the embeddings.

Feature Classification using SVM Classifier:

The FaceRecognition_SVM_Classifier.py file( Feature classification) takes a test image as input and performs face detection, feature extraction on it. 
It then predicts the test image to which actor/actress it  looks alike.
I have attached a test image but you can replace it by placing your image.

Link to compressed files :
This link contains facenet_keras.h5 file,compressed file of extracted faces and compressed file of face embeddings and the original dataset.
Link:	https://drive.google.com/drive/folders/1o7-0I3jkZbngGIs9NCjwnFkLRpj-QMW2?usp=sharing

Libraries to install:
	1.	NumPy
	2.	PIL
	3.	mtcnn
	4.	matplotlib
	5.	keras
	6.	sklearn
	
Things to remember: 
The compressed files, test images, the dataset and the python file should be in same directory. 
If it is in different directory then you need to specify the path in the  python file.


