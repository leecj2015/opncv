import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
import keras.backend as K
import os
import glob
import numpy as np
#diplay input data


def print_result(path):
    name_list=glob.glob(path)
    fig=plt.figure(figsize=(12,16))
    for i in range(3):
        img=Image.open(name_list[i])
        sub_img=fig.add_subplot(131+i)
        sub_img.imshow(img)
img_path='./img/superman/*'
in_path='./img/'
out_path='./output/'
name_list=glob.glob(img_path)
print(name_list)
print_result(img_path)
#make sure target_size
datagen=image.ImageDataGenerator()
gen_data=datagen.flow_from_directory(in_path,batch_size=1,shuffle=False,save_to_dir=out_path+'resize',save_prefix='gen',target_size=(244,244))
# for i in range(3):
# #     gen_data.next()
# #
# # print_result(out_path+'resize/*')

#anger rotaion
datagen=image.ImageDataGenerator(rotation_range=45)
gen=image.ImageDataGenerator()
data=gen.flow_from_directory(in_path,batch_size=1,class_mode=None,shuffle=True,target_size=(224,224))
np_data=np.concatenate([data.next()for i in range(data.n)])
for i in range(3):
    gen_data.next()
print_result(out_path+'rotation_range/*')
datagen=image.ImageDataGenerator(width_shift_range=0.3,height_shift_range=0.3)
gen=image.ImageDataGenerator()
data=gen.flow_from_directory(in_path,batch_size=1,shuffle=False,save_to_dir=out_path+'shift',save_prefix='gen',target_size=(224,224))
np_data=np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data=datagen.flow_from_directory(in_path,batch_size=1,shuffle=False,save_to_dir=out_path+'shift',save_prefix='gen',target_size=(224,224))
for i in range(3):
    gen_data.next()