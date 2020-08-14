# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 18:36:40 2020

@author: Alexandre Kleppa
"""


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#preprocessing

#training_set
#applying some transformations on  the training set to avoid overfitting on the training set
#image augmentation

train_datagenerator = ImageDataGenerator(rescale=1./255,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip= 'True',
                                         vertical_flip='True')

training_set = train_datagenerator.flow_from_directory('gastrointestinal_dataset/train_set',
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='binary')

#test_set

test_datagenerator = ImageDataGenerator(rescale=1./255)

test_set = test_datagenerator.flow_from_directory('gastrointestinal_dataset/test_set',
                              target_size = (64,64),
                              batch_size= 32,
                              class_mode='binary')

#building the CNN

from keras.models import Sequential
from keras import layers

#initializing the CNN
cnn = Sequential()

# add the covolutional layer
cnn.add(layers.Conv2D(filters=32,
               kernel_size=3,
               activation='relu',
               input_shape=[64,64,3]))

#add the pooling layer
cnn.add(layers.MaxPool2D(pool_size=2,
                         strides=2))

#add a second layer

cnn.add(layers.Conv2D(filters=32,
               kernel_size=3,
               activation='relu',
               input_shape=[64,64,3]))

#add the pooling layer
cnn.add(layers.MaxPool2D(pool_size=2,
                         strides=2))

#flattening 

cnn.add(layers.Flatten())

#full connection
cnn.add(layers.Dense(units=128,
                     activation='relu'))

#outputlayer
cnn.add(layers.Dense(units=8,
                     activation='softmax'))

#train the cnn
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(x = training_set, validation_data = test_set, epochs = 40)


#save model
#cnn.save('model')


# -------------------------------------trying it out with a only prediction
#from keras.models import load_model
# cnn=load_model('model')

from keras.preprocessing import image

test_image=image.load_img('gastrointestinal_dataset/test_set/esophagitis/esophagiti (1).jpg',target_size=(64,64))

#convert the image PIL -> array
test_image=image.img_to_array(test_image)
#convert the image into batch
test_image=np.expand_dims(test_image,axis=0)

result=cnn.predict(test_image)
#training_set.class_indices

if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'
        
print(prediction)



