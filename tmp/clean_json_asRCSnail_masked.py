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
 
#foldernames = glob.glob("2020*")
#output_foldername="cleaned"
foldernames = glob.glob("*2020*")
# output_foldername="cleaned_all_masked"
output_foldername="C:\Users\enlik\GitRepo\deltax\RCSnail-AI-lite\src\n1_m1"

data_counter = 0 #keeps track how many we have in folder already
random.shuffle(foldernames) #shuffling to get both directions mixed order, so if we take the tail as val set we get both
for rec_name in foldernames:
    print(foldernames[-3:])
    file_tag=rec_name
    if "val" in rec_name: 
        file_tag = rec_name[4:]

    with open(rec_name+'/'+file_tag+'.json') as json_file:
        data = json.load(json_file)
        #print(data)
        #print(type(data)) -- its a list of dicts

        previous_type = None #each line is of certain type - frame, measurements of other
        images_to_be_kept = []
        measurements_to_be_kept = []
        for i,line in enumerate(data):
            #print(line)
            data_type = line['t']
            if data_type not in ['F','D']:
                print("BIG NO NO, wrong data type",data_type)
                break
                
            if previous_type == data_type:
                #print(i, ": two of same type in a row", data_type)
                if data_type=="F": #we want to keep the earliest frame
                    pass
                if data_type=="D": #we want to keep the latest command
                   measurements_to_be_kept[-1] == [line['j']['s'],line['j']['t']] #replace last command with a later version of it

            else:
               if data_type=="F":
                   images_to_be_kept.append(line['j']['f'])
               if data_type=="D":
                   measurements_to_be_kept.append([line['j']['s'],line['j']['t']]) #steering and throttle
            

            previous_type = data_type

    print(len(images_to_be_kept))
    print(len(measurements_to_be_kept))

    measurements_to_be_kept = np.array(measurements_to_be_kept)#easier to operate with nparrays
     
    #crop to same lengt (recording might start with one or the other and end with same
    crop_len=min([len(images_to_be_kept),len(measurements_to_be_kept)])

    images_to_be_kept = images_to_be_kept[:crop_len] #these are file nrs
    measurements_to_be_kept=measurements_to_be_kept[:crop_len,:]





    for i,img_nr in enumerate(images_to_be_kept):
        filename = rec_name+'/images/frame-'+str(img_nr)+'.png'
        #image = io.imread(fname=filename)
        #should apply transformations here
        #print(type(image[0,0,0]),image[0,:10,:], image.shape)
        im = Image.open(filename)
        small = im.resize(size=[180,120], resample= Image.NEAREST)
        print(type(small))
        cropped = np.array(small)
        print(cropped.shape)
        cropped = cropped[-60:,:,:]
        cropped = cropped[:-10,:,:]
        for jj in range(33,50):
           dist=max(1,50-jj)
           print([jj,dist*2])
           cropped[jj,dist*2:-dist*2,:]=0
        if i%100==0:
            io.imsave("example"+str(data_counter+i)+".png",cropped)
        

        np.save(output_foldername+"/frame_n1_m1_"+str(data_counter+i).zfill(7)+".npy", cropped)
        np.save(output_foldername+"/commands_n1_m1_"+str(data_counter+i).zfill(7)+".npy", np.array(measurements_to_be_kept[i,:]))
    data_counter+=len(images_to_be_kept) #done with this folder, add the nr to counter
