from django.shortcuts import render, redirect
from sklearn.datasets import load_files       

from keras.utils import load_img, img_to_array, np_utils
from glob import glob
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from.  import extract_bottleneck_features
from.  import dogNames  
#from keras.preprocessing import image                  
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import os
from .models import ImageModel

import cv2                
import matplotlib.pyplot as plt                        




# Create your views here.
module_dir = os.path.dirname(__file__)   #get current directory
#file_path = os.path.join(module_dir, 'static/DogVGG16Data.npz')

'''#bottleneck_features = np.load(file_path)
train_res50 = bottleneck_features['train']
valid_res50 = bottleneck_features['valid']
test_res50 = bottleneck_features['test']'''

#face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def home(request):

    return render(request, 'home.html', {})

'''def res50_predict_breed(img_path):
    

    res50_model = Sequential()
    res50_model.add(GlobalAveragePooling2D(input_shape=train_res50.shape[1:]))
    
    res50_model.add(Dense(133, activation='softmax'))
    res50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    file_path = os.path.join(module_dir, 'static/weights.best.VGG16.hdf5')
    res50_model - keras.load_model(file_path)
    #res50_model = keras.models.load_model(file_path)
    bottleneck_feature =  extract_bottleneck_features.extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    print(bottleneck_feature.shape)
    
    predicted_vector = res50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    
    return dogNames.dog_names[np.argmax(predicted_vector)]'''

def Xception_predict_breed(img_path):
    '''
     predicts dog breed using pretrained Xeception model.  Model trained in jupyter notedbook
    '''

    file_path = os.path.join(module_dir, 'static/BreedXcept.h5')
    XceptModel = keras.models.load_model(file_path)
    tensor = path_to_tensor(img_path)
    prosTens = keras.applications.xception.preprocess_input(tensor)
    predictV  = XceptModel.predict(prosTens)
    return dogNames.dog_names[np.argmax(predictV)]

from matplotlib import pyplot as plt

from keras.applications.resnet import preprocess_input, decode_predictions, ResNet50

def ResNet50_predict_labels(img_path):
    ResNet50_model = ResNet50(weights='imagenet')
    tens = path_to_tensor(img_path)
    img = preprocess_input(tens)
    return np.argmax(ResNet50_model.predict(img))



'''def face_detector(imgData):
    
    facePath = os.path.join(module_dir, 'static/haarcascade_frontalface_alt.xml')
    face_cascade = cv2.CascadeClassifier(facePath)
   
    
    imgc = cv2.cvtColor(np.array(imgData), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0'''

def dog_detector(img_path):
    '''input: image to be classified
       output: whether image contains a dog or not
    '''
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def humanOrDogClass(img_path):
    '''
        classifies image as dog or human based on pretrained Xception model
    '''
    file_path = os.path.join(module_dir, 'static/Xface.h5')
    XfaceModel = keras.models.load_model(file_path)
    tensor = path_to_tensor(img_path)
    prosTens = keras.applications.xception.preprocess_input(tensor)
    predictV  = XfaceModel.predict(prosTens)
    list = ['Dog', 'Human']
    return list[np.argmax(predictV)]




def dogHumanClassifier(img_path):
    '''input: image

       output: classfies human or dog, then finds breed of best resemblance  
    
    '''
    facePath = os.path.join(module_dir, 'static/Xface.h5')

    #dog = dog_detector(img_path)
    #human  = face_detector(img_path)
     
    human_dog = humanOrDogClass(img_path)
    
    if human_dog == 'Human':
        breed = Xception_predict_breed(img_path)
        print('hello human, you like a '+breed)
        return breed, "hello, human, this is the dog breed you look like: "
        #print('no human or dog')
        #return 'no human or dog'
        
    elif human_dog == 'Dog':
       
        breed = Xception_predict_breed(img_path)
        print('hello doggy, you are a '+breed)
        return breed, "hello, doggo, this is your breed: "
    
        
        


def Import(request):
    print(request.method)

    '''upload image for classification and send result to html page'''

    if request.method  == 'POST':
        #form = SaveFileForm(request.POST, request.FILES)
        image = request.FILES['image']
        ImageModel.objects.all().delete()
        imageM = ImageModel.create(image)
        imageM.save()
        
        image2 = ImageModel.objects.all().first() 
        
        imgP = Image.open(imageM.Image)
        
        breed, response = dogHumanClassifier(imgP)
        print(breed)
        breed = breed[15:].replace('_', ' ')
        print(image2.Image.url)
        
        
        return render(request, 'home.html', {'breed': breed, 'response': response, 'image1': image2})
    
    else:
        print("this worked")
        return render(request, 'home.html', {})
    
def path_to_tensor(img1):

    '''input: image 
       output: numpy tensor

       converts image to numpy tensor
    '''
    # loads RGB image as PIL.Image.Image type
    #file_path = os.path.join(module_dir, img_path)
    
    img1 = img1.resize((224,224))
    #img = image.load_img( img1, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x1 = image.img_to_array(img1)
    print(x1.shape)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    finalX = np.expand_dims(x1, axis=0)
    print(finalX.shape)
    return finalX

def paths_to_tensor(img_paths):
    '''converts multiple images to numpy tensor'''
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

