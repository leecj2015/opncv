import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from  PIL import Image
from keras.preprocessing import image
import  os
import glob
import keras.backend as K

def print_result(path):
    name_list=glob.glob(path)
    fig=plt.figure(figsize=(12,16))
    for i in range(len(name_list)):
        img=Image.open(name_list[i])
        sub_img=fig.add_subplot(131+i)
        sub_img.imshow(img)


img_path='./img/Superman/*'
in_path='./img'
out_path='./output/'
name_list=glob.glob(img_path)
print(name_list)
print_result(img_path)
plt.show()

#指定target_size后所有图像变为相同大小
datagen=image.ImageDataGenerator()
gen_data=datagen.flow_from_directory(in_path,batch_size=1,shuffle=False,
                                     save_to_dir=out_path+'resize',
                                     save_prefix='gen',
                                     target_size=(224,224))
# if not os.path.join(out_path + 'resize') is exit():
#     os.mkdir(out_path + 'resize')
for i in range(3):
    gen_data.next()
print_result(out_path+'resize/*')


#角度旋转
datagen=image.ImageDataGenerator(rotation_range=45)
gen=image.ImageDataGenerator()
data=gen.flow_from_directory(in_path,batch_size=1,class_mode=None,shuffle=True,target_size=(224,224))
np_data=np.concatenate([data.next()for i in range(data.n)])
datagen.fit(np_data)
gen_data=datagen.flow_from_directory(in_path,batch_size=1,shuffle=False,save_to_dir=out_path+'rotation_range',save_prefix='gen',target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path+'rotation_range/*')

#shift
datagen=image.ImageDataGenerator(width_shift_range=0.3,height_shift_range=0.3)
gen=image.ImageDataGenerator()
data=gen.flow_from_directory(in_path,batch_size=1,class_mode=None,shuffle=True,target_size=(224,224))
np_data=np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data=datagen.flow_from_directory(in_path,batch_size=1,shuffle=False,save_to_dir=out_path+'shift_range',save_prefix='gen',target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path+'shift_range/*')
#zoom
datagen=image.ImageDataGenerator(zoom_range=0.5)
gen=image.ImageDataGenerator()
data=gen.flow_from_directory(in_path,batch_size=1,class_mode=None,shuffle=True,target_size=(224,224))
np_data=np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data=datagen.flow_from_directory(in_path,batch_size=1,shuffle=False,save_to_dir=out_path+'zoom',save_prefix='gen',target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path+'zoom/*')

#channel_shift
datagen=image.ImageDataGenerator(channel_shift_range=20)
gen=image.ImageDataGenerator()
data=gen.flow_from_directory(in_path,batch_size=1,shuffle=True,class_mode=None,target_size=(224,224))
np_data=np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data=datagen.flow_from_directory(in_path,batch_size=1,shuffle=False,save_to_dir=out_path+'channel_shift',save_prefix='gen',target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path+'channel_shift/*')

#horizontal_flip
datagen=image.ImageDataGenerator(horizontal_flip=True)
gen=image.ImageDataGenerator()
data=gen.flow_from_directory(in_path,batch_size=1,class_mode=None,shuffle=True,target_size=(224,224))
np_data=np.concatenate([data.next()for i in range(data.n)])
datagen.fit(np_data)
gen_data=gen.flow_from_directory(in_path,batch_size=1,shuffle=False,save_to_dir=out_path+'horiza_range',save_prefix='gen',target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path+'horiza_range/*')


#wrap
datagen=image.ImageDataGenerator(fill_mode='wrap',zoom_range=[4,4])
gen=image.ImageDataGenerator()
data=gen.flow_from_directory(in_path,batch_size=1,class_mode=None,shuffle=True,target_size=(224,224))
np_data=np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data=datagen.flow_from_directory(in_path,batch_size=1,shuffle=False,save_prefix='gen',save_to_dir=out_path+'wrap_range',target_size=(224,224))
for i in range(3):
    gen_data.next()
print_result(out_path+'wrap_range/*')