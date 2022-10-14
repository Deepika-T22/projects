#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import os
import cv2
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance, ImageChops, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from keras import models
from keras import layers
from keras.layers import Dense,GlobalAveragePooling2D,Flatten,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras import optimizers
from keras.models import Model
from keras.models import Sequential


# In[2]:


model=Sequential()
# filters filtersize padding input
model.add(Conv2D(32, (3, 3), padding="same",input_shape=(100,100,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation("softmax"))
model.summary()


# In[3]:


folders=glob('C:\\Users\\mini_project\\dataset\\testing\\*')


# In[4]:


print(folders)


# In[5]:


x=Flatten()(model.output)


# In[6]:


prediction=Dense(len(folders),activation='softmax')(x)
model=Model(inputs=model.input,outputs=prediction)


# In[7]:


model.summary()


# In[8]:


data_gen_train = ImageDataGenerator(rescale=1/255.)

data_gen_valid = ImageDataGenerator(rescale=1/255.)

train_generator = data_gen_train.flow_from_directory('C:\\Users\\mini_project\\dataset\\training', target_size=(100,100),batch_size=16, class_mode='categorical')

valid_generator = data_gen_valid.flow_from_directory('C:\\Users\\mini_project\\dataset\\testing', target_size=(100,100),batch_size=16, class_mode='categorical')


# In[9]:


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[10]:


history=model.fit(train_generator,validation_data=valid_generator,epochs=16)


# In[11]:


model.save("cnn.h5")


# In[ ]:





# In[ ]:




