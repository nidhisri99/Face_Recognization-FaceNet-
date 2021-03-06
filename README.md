# Face_Recognization-FaceNet-

This is simple real Time face recognization using FaceNet. FaceNet provides a unified embedding for face recognition, verification and clustering tasks. It maps each face image into a euclidean space such that the distances in that space correspond to face similarity, i.e. an image of person A will be placed closer to all the other images of person A as compared to images of any other person present in the dataset.</br>
Here in this face recognition system, we are able to recognize a person’s identity by just feeding one picture of that person’s face to the system. And, in case, it fails to recognize the picture, it means that this person’s image is not stored in the system’s database.</br>
    To solve this problem, we cannot use only a convolutional neural network for two reasons:</br>
    1) CNN doesn’t work on a small training set. </br>
    2) It is not convenient to retrain the model every time we add a picture of a new person to the system. </br>
However, we can use Siamese neural network for face recognition.

<h2>Siamese neural network</h2>
  Siamese neural network has the objective to find how similar two comparable things are (e.g. signature verification, face recognition..). This network has two identical subnetworks, which both have the same parameters and weights. 
  
  ![alt text](https://miro.medium.com/max/700/1*ZQjqmkyFyAQW34KIA26uwQ.png)
  
  ### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning. OR You can use CMD or terminals.
```
pip install requirements.txt
```

### Download the pretrained weights
* Download [this](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view) pretrained Facenet model and copy to model folder.
* Download [this](https://github.com/wangbm/MTCNN-Tensorflow/tree/master/save_model) pretrained MTCNN models and copy to mtcnn_model.


### Procedure
1. Download the code or git clone https://github.com/nidhisri99/Face_Recognization-FaceNet-.git.
2. Add your images to the "images" folder. (Only one image is enough)
3. Run face_recognition.py. </br>
To run use command.
```
python face_recognition.py
```

After running your face_recognition.py it should look like this.
![Alt text](git_images/face_recognition.png?raw=true "Title")

Examples</br>
If the face is found in the images folder.</br>
![Alt text](git_images/face1.png?raw=true "Title")
</br>
</br>
If face is not found.</br>
</br>
![Alt text](git_images/no_match.png?raw=true "Title")
