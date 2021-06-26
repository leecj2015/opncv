'''
为了高雄啊存储，可将数据进行序列化存储，这样也便于网络流式读取数据，TFRecord是一种
常用的存储二进制序列数据的方法
tf.Example类是一种将数据表示维{“string”:value}形式message类型，Tensorflow使用tf.ExampleL来吸入，
tf.ByteList 可以使用类型包括string和byte
tf.FloatList:
tf.Int64List:可以用于enum,bool,int32,uint32,int64
'''

import tensorflow as tf
import numpy as np
def _bytes_feature(value):
    '''Return a bytes_list from a string/byte'''
    if isinstance(value,type(tf.constant(0))):
        value=value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    '''returen a float_list from a float/double'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
    '''returen a float_list from a float/double'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#tf.train.BytesList
print(_bytes_feature(b'test_string'))
print(_bytes_feature('test_string'.encode('utf8')))

#tf.train.FloatList
print(_float_feature(np.exp(1)))

def serialize_example(feature0,feature1,feature2,feature3):
    '''
    create tf.Example
    :param feature0:
    :param feature1:
    :param featrue2:
    :param feature3:
    :return:
    '''
    feature={
        'feature0':_int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }

    #use tf.train.Example
    example_proto=tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
#data
n_observations=int(1e4)
print(n_observations)
#boolean feature
feature0=np.random.choice([False,True],n_observations)
#interger feature
feature1=np.random.randint(0,5,n_observations)
#string feature
strings=np.array([b'cat',b'dog',b'chicken'])
feature2=strings[feature1]
#float feature
features3=np.random.randn(n_observations)
filename='tfrecord-1'
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example=serialize_example(feature0[i],feature1[i],feature2[i],feature3[i])
        writer.write(example)


#加载数据
filenames=[filename]
#读取
raw_dataset=tf.data.TFRecordDataset(filenames)
print(raw_dataset)