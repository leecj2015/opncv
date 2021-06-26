import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
model=keras.models.load_model('fashion_model.h5')
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
def load_data(data_folder):

  files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]

  paths = []
  for fname in files:
    paths.append(os.path.join(data_folder,fname))

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)
(train_images, train_labels), (test_images, test_labels) = load_data('./data/fashion/')
predictions=model.predict(test_images)
b=predictions.shape
print(b)
def plot_image(i,predicitons_array,true_label,img):
    predicitons_array,true_label,img=predicitons_array,true_label[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,cmap=plt.cm.binary)
    predicted_label=np.argmax(predicitons_array)
    if predicted_label==true_label:
        color='blue'
    else:
        color='red'
    plt.xlabel('{} {}'.format(class_names[predicted_label],
                              100*np.max(predicitons_array),
                              class_names[true_label],
                              color=color))
def plot_value_array(i,precitions_arry,true_label):
    precitions_arry,true_label=precitions_arry,true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot=plt.bar(range(10),precitions_arry,color='#777777')
    plt.ylim([0,1])
    predicted_labe=np.argmax(precitions_arry)
    thisplot[predicted_labe].set_color('red')
    thisplot[true_label].set_color('blue')
num_rows=5
num_cols=3
num_images=num_cols*num_cols
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_image(i,predictions[i],test_labels,test_images)
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plot_value_array(i,predictions[i],test_labels)
plt.tight_layout()
plt.show()

#注意测试时候，需要对数据进行形同的预处理
train_images=train_images/255.0
test_images=test_images/255.0
predictions=model.predict(test_images)
num_rows=5
num_cols=3
num_images=num_cols*num_cols
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_image(i,predictions[i],test_labels,test_images)
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plot_value_array(i,predictions[i],test_labels)
plt.tight_layout()
plt.show()