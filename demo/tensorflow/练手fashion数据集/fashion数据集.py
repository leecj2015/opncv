# from tensorflow.examples.tutoriials.mnist import input_data
# data=input_data.read_data_sets('data/fashion')
# fashion_mnist=keras.datasets.fashion_mnist
import numpy as np
import os
import gzip
#定义加载函数，data_folder为保存gz数据的文件夹，该文件下有4个文件夹
# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
# 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
import numpy as np
import os
import gzip
import tensorflow as tf
from tensorflow import keras
# 定义加载数据的函数，data_folder为保存gz数据的文件夹，该文件夹下有4个文件
# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
# 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'

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

class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
import matplotlib.pyplot as plt
# 创建一个新图形
# plt.figure()
# # 显示一张图片在二维的数据上 train_images[0] 第一张图
# plt.imshow(train_images[0])
# # 在图中添加颜色条
# plt.colorbar()
# # 是否显示网格线条,True: 显示，False: 不显示
# plt.grid(False)
# plt.show()

#在给神经网络提供模型之前，将其值缩放到0-1范围，为此，将图像组件的数据类型从整数转换成浮点数，并除以255.变成预处理函数
#训练图像缩放在0-1范围内
train_images=train_images/255.0
#测试图像缩放
test_images=test_images/255.0
'''
显示来自训练集的前25个图像，并在每个图像下面显示类名。
验证数据的格式是否正确，我们准备构建和训练网络。
'''
#保存画布的图像，长度、宽度都为10
plt.figure(figsize=(10,10))
#显示训练集的25张图像
for i in range(25):
    #创建分布5*5个图形
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    #显示照片，以CM为单位
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''
建立模型
当像素处理后，神经网络由两个 tf.keras.layer 序列组成。致密层。
它们是紧密相连的，或者说完全相连的神经层。第一个致密层有128个节点
(或神经元)。第二个(也是最后一个)层是10个节点的 softmax 层——
它返回一个10个概率分数的数组，其和为1。每个节点包含一个分数，表示
当前图像属于这10个类之一的概率。
'''

#建立模型
def build_model():
    #线性叠加
    model=tf.keras.models.Sequential()
    #改版平缓输入
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    #第一层紧密连接128神经元
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    #第二层分10个类别
    model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
    return  model
#compile model
model=build_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
'''
optimizer:   模型如何更新基于它看到的数据和它的损失函数。tf.train.AdamOptimizer(), 使用Adam梯度下降优化方法，降低损失。
loss : 用来测量模型在训练过程中的精确度。最小化这个函数来“引导”模型向正确的方向。‘sparse_categorical_crossentropy’ : 稀疏多分类交叉熵损失。
metrics: 用于监视培训和测试步骤。使用了 'accuracy'，精确分类的图像的比例。
API接口描述

compile(self,

        optimizer,

        loss=None,

        metrics=None,
 
        loss_weights=None,

        sample_weight_mode=None,

        weighted_metrics=None,

        target_tensors=None,

        distribute=None,

        **kwargs):

作用：配置参数，主要用于训练的模型

参数描述：

optimizer: optimizer的名称字符串，或者optimizer的类。
loss: 损失目标函数的名称字符串，或者目标函数
metrics: 模型评估的指标列表，在评估训练和测试模型。一般 metrics = ['accuracy']
loss_weights: 标量的列表或字典，系数（Python 浮点数）衡量不同模型的损失权重。
sample_weight_mode: 样品权重模式
weighted_metrics: 训练和测试模型期间，将根据样本权重或类权重进行评估和加权的指标列表
target_tensors: keras为模型的目标值创建占位符，在训练期间，fed目标数据。
distribute: 我们想要用来分发模型训练(DistributionStrategy实例类)
**kwargs: 这些参数转递给会话 'tf.Session.run'
'''
#训练模型
model.fit(train_images,train_labels,epochs=10)
'''
fit (self,

    x=None,

    y=None,

    batch_size=None,

    epochs=1,

    verbose=1,

    callbacks=None,

    validation_split=0.,

    validation_data=None,

    shuffle=True,

    class_weight=None,

    sample_weight=None,

    initial_epoch=0,

    steps_per_epoch=None,

    validation_steps=None,

    max_queue_size=10,

    workers=1,

    use_multiprocessing=False,

    **kwargs):

作用：指定数量次数迭代（数据集依次迭代）训练模型
主要介绍下面部分参数：

x: 输入数据，特征
y: 目标数据，标签
batch_size: 每次梯度更新的样本数量大小
epochs: 训练模型的次数
verbose: 冗长的模式, 0,1,或者 2; 0表示无，1表示进度显示, 2表示每次一行
callbacks: 'keras.callbacks.Callback'的类，训练模型时的回调函数列表
'''
#评估模型
#比较测试集与训练集这两个的情况，是否会出现过拟合或者欠拟合的情况
test_loss,test_acc=model.evaluate(test_images,test_labels)
print('测试损失：%f 测试准确率：%f'%(test_loss,test_acc))
'''
evaluate( self,

         x=None,

         y=None,

         batch_size=None,

         verbose=1,

         sample_weight=None,

         steps=None,

         max_queue_size=10,

         workers=1,

         use_multiprocessing=False):

作用：返回模型下的测试数据集的损失值和准确率，评估模型
主要介绍下面部分参数：

x: 测试特征数据集
y: 测试标签数据集
batch_size:  每次梯度更新的样本数量大小
verbose: 冗长的模式, 0,1,或者 2; 0表示无，1表示进度显示, 2表示每次一行
sample_weight: 测试样本权重的numpy数组
steps: 步长总数
'''
#使用模型做预测
predictions=model.predict(test_images)
#提取20个数据集，进行预测判断是否正确
for i in range(25):
    pre=class_names[np.argmax(predictions[i])]
    tar=class_names[test_labels[i]]
    print('预测：%s 实际：%s'%(pre,tar))
#在图像中画出
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt .yticks([])
    plt.grid(False)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    #预测的图片是否正确，黑色底表示预测正确，红色底表示预测错误
    predited_label=np.argmax(predictions[i])
    true_label=test_labels[i]
    if predited_label==true_label:
        color='black'
    else:
        color='red'
    plt.xlabel('{} {}'.format(class_names[predited_label],class_names[true_label],
                              color=color))
plt.show()

#保存训练好的模型
#保存权重参数与网络模型
model.save('fashion_model.h5')
config=model.to_json()
print(config)
with open('config_json')as json:
    json.write(config)
model=keras.models.model_from_json(json_config)
model.summary()
#权重参数
weights=model.get_weights()
model.save_weights('weights.h5')
model.load_weights('weights.h5')
