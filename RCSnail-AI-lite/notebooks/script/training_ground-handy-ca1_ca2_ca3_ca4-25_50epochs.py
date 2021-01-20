#!/usr/bin/env python
# coding: utf-8

from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 75%; }
</style>
"""))


get_ipython().system('dir')


from platform import python_version

print(python_version())


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
                                


def plot_stuff(title, plot_elem, figsize=(18, 10)):
    fig=plt.figure(figsize=figsize)
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #x = np.arange(0, len(plot_elems[0]['data']), 1)
    
    #for plot_elem in plot_elems:
    #    plt.errorbar(x, plot_elem['data'], yerr=plot_elem['error'], label=plot_elem['label'], alpha=plot_elem['alpha'], fmt='-o', capsize=5)
    
    plt.plot(list(range(1,len(plot_elem['data'])+1)), plot_elem['data'])
    plt.grid(axis='both')
    #plt.legend(loc='best', prop={'size': 15})
    plt.show()
    plt.savefig('./' + title + '.png')


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
val_data_loc="../cleaned_all/"
filenames = glob.glob("../cleaned_all/*")
print(filenames[1])
print(int(len(filenames)/2))
nr_of_datapoints = int(len(filenames)/2) #label and image files
print(nr_of_datapoints)


MAEs=[]
preds=[]
labels=[]


frames = []
commands = []

# for batch in range(53482,53482+nr_of_datapoints): # using the end of file. 32 batches of size batch of 32
for batch in range(1,nr_of_datapoints): # using the end of file. 32 batches of size batch of 32
    frames.append(np.load(val_data_loc + "frame_"+str(batch).zfill(7)+".npy"))
    commands.append(np.load(val_data_loc + "commands_"+str(batch).zfill(7)+".npy"))


#frames = frames.reshape(1,60,180,3)
frames = np.array(frames)
frames.shape


#commands = commands.reshape(1, 2)
commands = np.array(commands)
commands.shape


# ### Fitting with 25 Epochs

model = create_standalone_nvidia_cnn(activation='linear', input_shape=(60, 180, 3), output_shape=2)
model.summary()


hist = model.fit(frames, commands, batch_size=64, epochs=25, validation_split=0.2)

losses = []
val_losses = []
current_loss = hist.history['loss']
current_val_loss = hist.history['val_loss'] 

losses.append(current_loss)
print(val_losses)
val_losses.append(current_val_loss)

tqdm.write("Loss per epoch: {}".format(current_loss))
tqdm.write("Validation loss per epoch: {}".format(current_val_loss))

gc.collect()


print(val_losses)
print(hist.history['val_loss'] )
loss_data = []
val_loss_data = []


val_loss_data = {'data': hist.history['val_loss'], 'label': 'Validation loss', 'alpha': 1.0}
plot_stuff('Nivida cnn standalone validation loss', val_loss_data, figsize=(10, 6))


loss_data = {'data': losses[0], 'label': 'Loss', 'alpha': 1.0}
plot_stuff('Training', loss_data, figsize=(10, 6))


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
model.save('../src/model_team_3_' + now + '_25epochs_ca1_ca2_ca3_ca4.h5')


#todo load test set into memeory, evaluate
import keras
model = keras.models.load_model('../src/model_team_3_' + now + '_25epochs_ca1_ca2_ca3_ca4.h5')
import glob

val_data_loc="../cleaned_all"
filenames = glob.glob("../cleaned_all/*")
print(len(filenames))
nr_of_datapoints = int(len(filenames)/2) #label and image files


MAEs=[]
preds=[]
labels=[]

# for batch in range(53482//32,(53482+nr_of_datapoints)//32):
for batch in range(nr_of_datapoints//32-32,nr_of_datapoints//32):
# for batch in range(nr_of_datapoints//32-32,nr_of_datapoints//32): # using the end of file. 32 batches of size batch of 32
    frames=np.zeros((32,60,180,3))
    commands = np.zeros((32,2))
    for i in range(32):
#         frames.append(np.load(val_data_loc + "frame_"+str(batch).zfill(7)+".npy"))
#         commands.append(np.load(val_data_loc + "commands_"+str(batch).zfill(7)+".npy"))
#         frames[i,:] = np.load("../cleaned_all/frame_"+str(batch*32+i).zfill(7)+".npy")
#         commands[i] = np.load("../cleaned_all/commands_"+str(batch*32+i).zfill(7)+".npy")
        frames[i,:] = np.load("../cleaned_all/frame_"+str(batch).zfill(7)+".npy")
        commands[i] = np.load("../cleaned_all/commands_"+str(batch).zfill(7)+".npy")
    MAEs.append(model.evaluate(frames,commands, batch_size=32))
    pred = model.predict(frames)
    preds.append(pred)
    labels.append(commands)
    
print(np.mean(MAEs))


print(np.mean(MAEs))
p_steer=np.array(preds)[:,:,0]
l_steer=np.array(labels)[:,:,0]
p_throttle=np.array(preds)[:,:,1]
l_throttle=np.array(labels)[:,:,1]

print(p_steer.shape,l_steer.shape)
p_steer=p_steer.flatten()
l_steer=l_steer.flatten()
p_throttle=p_throttle.flatten()
l_throttle=l_throttle.flatten()


plt.figure(figsize=(32,8))
plt.plot(range(1000),l_steer[:1000],label="human steering")
plt.plot(range(1000),p_steer[:1000],label="predicted")
plt.legend()
plt.show()


plt.figure(figsize=(32,8))
plt.plot(range(1000),l_throttle[:1000],label="human throttle")
plt.plot(range(1000),p_throttle[:1000],label="predicted")
plt.legend()
plt.show()



plt.scatter(l_steer,p_steer,s=0.001)


# ### Fitting with 50 Epochs

model2 = create_standalone_nvidia_cnn(activation='linear', input_shape=(60, 180, 3), output_shape=2)
model2.summary()


hist = model2.fit(frames, commands, batch_size=64, epochs=50, validation_split=0.2)

losses = []
val_losses = []
current_loss = hist.history['loss']
current_val_loss = hist.history['val_loss'] 

losses.append(current_loss)
print(val_losses)
val_losses.append(current_val_loss)

tqdm.write("Loss per epoch: {}".format(current_loss))
tqdm.write("Validation loss per epoch: {}".format(current_val_loss))

gc.collect()


print(val_losses)
print(hist.history['val_loss'] )
loss_data = []
val_loss_data = []


val_loss_data = {'data': hist.history['val_loss'], 'label': 'Validation loss', 'alpha': 1.0}
plot_stuff('Nivida cnn standalone validation loss', val_loss_data, figsize=(10, 6))


loss_data = {'data': losses[0], 'label': 'Loss', 'alpha': 1.0}
plot_stuff('Training', loss_data, figsize=(10, 6))


mem_frame = frames[10].reshape(1,60,180,3)
mem_frame.shape
mem_frame

new_mem_frame = mem_frame

if(new_mem_frame.all() == mem_frame.all()):
    print("sama")


model2.predict(mem_frame)


from datetime import datetime

# Get current timestamp | source: https://www.programiz.com/python-programming/datetime/current-datetime
now = datetime.now().strftime("%Y%m%d_%H%M%S")
print(now)

#plot_model(model, to_file=model_path + model_file_prefix + model_file_suffix.format(model_number, 'png'), show_shapes=True)
model2.save('../src/model_team_3_' + now + '_50epochs_ca1_ca2_ca3_ca4.h5')


#todo load test set into memeory, evaluate
import keras
model = keras.models.load_model('../src/model_team_3_' + now + '_50epochs_ca1_ca2_ca3_ca4.h5')
import glob

val_data_loc="../cleaned_all"
filenames = glob.glob("../cleaned_all/*")
print(len(filenames))
nr_of_datapoints = int(len(filenames)/2) #label and image files


MAEs=[]
preds=[]
labels=[]

# for batch in range(53482//32,(53482+nr_of_datapoints)//32):
for batch in range(nr_of_datapoints//32-32,nr_of_datapoints//32):
# for batch in range(nr_of_datapoints//32-32,nr_of_datapoints//32): # using the end of file. 32 batches of size batch of 32
    frames=np.zeros((32,60,180,3))
    commands = np.zeros((32,2))
    for i in range(32):
#         frames.append(np.load(val_data_loc + "frame_"+str(batch).zfill(7)+".npy"))
#         commands.append(np.load(val_data_loc + "commands_"+str(batch).zfill(7)+".npy"))
#         frames[i,:] = np.load("../cleaned_all/frame_"+str(batch*32+i).zfill(7)+".npy")
#         commands[i] = np.load("../cleaned_all/commands_"+str(batch*32+i).zfill(7)+".npy")
        frames[i,:] = np.load("../cleaned_all/frame_"+str(batch).zfill(7)+".npy")
        commands[i] = np.load("../cleaned_all/commands_"+str(batch).zfill(7)+".npy")
    MAEs.append(model.evaluate(frames,commands, batch_size=32))
    pred = model.predict(frames)
    preds.append(pred)
    labels.append(commands)
    
print(np.mean(MAEs))


print(np.mean(MAEs))
p_steer=np.array(preds)[:,:,0]
l_steer=np.array(labels)[:,:,0]
p_throttle=np.array(preds)[:,:,1]
l_throttle=np.array(labels)[:,:,1]

print(p_steer.shape,l_steer.shape)
p_steer=p_steer.flatten()
l_steer=l_steer.flatten()
p_throttle=p_throttle.flatten()
l_throttle=l_throttle.flatten()


plt.figure(figsize=(32,8))
plt.plot(range(1000),l_steer[:1000],label="human steering")
plt.plot(range(1000),p_steer[:1000],label="predicted")
plt.legend()
plt.show()


plt.figure(figsize=(32,8))
plt.plot(range(1000),l_throttle[:1000],label="human throttle")
plt.plot(range(1000),p_throttle[:1000],label="predicted")
plt.legend()
plt.show()



plt.scatter(l_steer,p_steer,s=0.001)




