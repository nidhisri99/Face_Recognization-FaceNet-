import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,glob
import mtcnn
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from keras_facenet import FaceNet
from scipy.spatial import distance
#%%
detector = MTCNN()
def extract_face(img, required_size=(160, 160)):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img)
    x1, y1, w, h = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h
    # extract the face
    face = img[y1:y2, x1:x2]
    # resize pixels to the model size
    image = cv2.resize(face,required_size)
    return image
#%%
x=[]
#ext = ['png', 'jpg', 'jpeg', 'gif']
#files = glob.glob(r'D:\IEEE paper\deep learning completed project\srinidhiface\images\*\*.jpeg') + glob.glob(r'D:\IEEE paper\deep learning completed project\srinidhiface\images\*\*.jpg')
for i in glob.glob(r'D:\IEEE paper\deep learning completed project\srinidhiface\images\*\*.jpeg'):
    img = cv2.imread(i)
    img=extract_face(img)
    x.append(img)
x=np.stack(x)    
#%%
embedder = FaceNet()
embeddings = embedder.embeddings(x)   
#%%
a=[]
for root,dirp,file in os.walk(r'D:\IEEE paper\deep learning completed project\srinidhiface\images'):
    a.append(dirp)
#%%
dictq={}
for i in range(len(a[0])):
    dictq[a[0][i]]=embeddings[i]
print(dictq)    
#%%
for key,value in dictq.items():
    value.shape=[512,1]
#%%    
b=[]
cap = cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()

    #cv2.imshow('face',frame)

    img1=extract_face(frame)
    plt.imshow(frame)
    img1=np.expand_dims(img1,axis=0)
    emb=embedder.embeddings(img1)
    emb=np.transpose(emb)
    min_dist=100
    for key,value in dictq.items():
        dist=np.linalg.norm(emb-value)
        b.append(dist)
        if dist<min_dist:
            min_dist=dist
            identity=key
    if min_dist < 0.6:
       cv2.putText(frame, "Face : " + identity,(100,100),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0,255), 2)
    else:
       cv2.putText(frame,'no match',(100,100),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),2)
    cv2.imshow('face',frame)

    if cv2.waitKey(1) &    0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
#%%
