#!/usr/bin/env python
# coding: utf-8

from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 75%; }
</style>
"""))


get_ipython().system('dir')


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
import gc
import time
from tqdm.notebook import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

module_path_list = [os.path.abspath(os.path.join('../')), 
                    os.path.abspath(os.path.join('../../RCSnail-Commons'))]

for module_path in module_path_list:
    if module_path not in sys.path:
        sys.path.append(module_path)

from commons.configuration_manager import ConfigurationManager
#from src.utilities.transformer import Transformer
from src.learning.training.generator import Generator, GenFiles
#from src.learning.models import create_standalone_nvidia_cnn, create_standalone_resnet
                                


def create_standalone_nvidia_cnn(activation='linear', input_shape=(60, 180, 3), output_shape=1):
    """
    Activation: linear, softmax.
    Architecture is from nvidia paper mentioned in https://github.com/tanelp/self-driving-convnet/blob/master/train.py
    """
    from tensorflow.keras.layers import Convolution2D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

    inputs = Input(shape=input_shape)
    conv_1 = Convolution2D(24, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="elu")(inputs)
    conv_2 = Convolution2D(36, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="elu")(conv_1)
    conv_3 = Convolution2D(48, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="elu")(conv_2)
    conv_4 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), padding="same", activation="elu")(conv_3)
    conv_5 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), padding="same", activation="elu")(conv_4)
    flatten = Flatten()(conv_5)
    dense_1 = Dense(1164, kernel_regularizer=l2(0.0005), activation="elu")(flatten)
    dense_2 = Dense(100, kernel_regularizer=l2(0.0005), activation="elu")(dense_1)
    dense_3 = Dense(50, kernel_regularizer=l2(0.0005), activation="elu")(dense_2)
    dense_4 = Dense(10, kernel_regularizer=l2(0.0005), activation="elu")(dense_3)
    out_dense = Dense(output_shape, activation=activation)(dense_4)

    model = Model(inputs=inputs, outputs=out_dense)
    optimizer = Adam(lr=3e-4)
    model.compile(loss=mean_absolute_error, optimizer=optimizer)

    return model


# # Handy Testing

import json
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
import skimage
from skimage.transform import rescale, resize, downscale_local_mean

import glob
import random
from PIL import Image
import PIL
 


import glob

# val_data_loc="preprocessed/cleaned_all"
# filenames = glob.glob("preprocessed/cleaned_all/*")
val_data_loc="../n1_m1/"
filenames = glob.glob("../n1_m1/*")
print(filenames[1])
print(int(len(filenames)/2))
nr_of_datapoints = int(len(filenames)/2) #label and image files
print(nr_of_datapoints)


MAEs=[]
preds=[]
labels=[]


frames = []
commands = []
for batch in range(1,nr_of_datapoints): # using the end of file. 32 batches of size batch of 32
    #frames=np.zeros((1,60,180,3))
    #commands = np.zeros((1,2))
    
#     frames.append(np.load("preprocessed/cleaned_all/frame_"+str(batch).zfill(7)+".npy"))
#     commands.append(np.load("preprocessed/cleaned_all/commands_"+str(batch).zfill(7)+".npy"))
    frames.append(np.load(val_data_loc + "frame_n1_m1_"+str(batch).zfill(7)+".npy"))
    commands.append(np.load(val_data_loc + "commands_n1_m1_"+str(batch).zfill(7)+".npy"))
    
    
    #print(commands)
    
    
    
    #MAEs.append(model.evaluate(frames,commands, batch_size=32))
    #pred = model.predict(frames)
    #preds.append(pred)
    #labels.append(commands)


#frames = frames.reshape(1,60,180,3)
frames = np.array(frames)
frames.shape


#commands = commands.reshape(1, 2)
commands = np.array(commands)
commands.shape


model = create_standalone_nvidia_cnn(activation='linear', input_shape=(60, 180, 3), output_shape=2)
model.summary()


model.fit(frames, commands, batch_size=64, epochs=10, validation_split=0.2)


mem_frame = frames[10].reshape(1,60,180,3)
mem_frame.shape
mem_frame

new_mem_frame = mem_frame

if(new_mem_frame.all() == mem_frame.all()):
    print("sama")


model.predict(mem_frame)


from datetime import datetime

# Get current timestamp | source: https://www.programiz.com/python-programming/datetime/current-datetime
now = datetime.now().strftime("%Y%m%d_%H%M%S")
print(now)

#plot_model(model, to_file=model_path + model_file_prefix + model_file_suffix.format(model_number, 'png'), show_shapes=True)
model.save('../src/model_team_3_' + now + '.h5')




