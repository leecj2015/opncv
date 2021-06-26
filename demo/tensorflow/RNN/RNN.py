import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import pprint
import logging
import time
from collections import Counter
from pathlib import Path
from tqdm import tqdm
#加载影评数据集，可以手动下载放到对应位置
(x_train,y_train),(x_text,y_test)=tf.keras.datasets.imdb.load_data()
print(x_train.shape)
#都进来的数据已经转化成ID隐射的，一般数据都进去都是词语，都需要动手转换成ID映射
print(x_train[0])
#词和ID的映射表，
with open('./imdb_word_index.json',encoding='utf-8')as f:
    for line in f:# _word2idx=tf.keras.datasets.imdb.get_word_index()
        word2idx={w:i+3 for w,i in line.items()}
        word2idx['<pad>']=0
        word2idx['<start>']=1
        word2idx['<unk>']=2
        idx2word={i:w for w,i in word2idx.items()}
#按文本大小进行排序
def sort_by_len(x,y):
    x,y=np.array(x),np.array(y)
    idx=sorted(range(len(x)),key=lambda i:len(x[i]))
#将中间结果保存到本地，万一程序崩溃还可以进行，保存的是文本数据，不是ID
x_train,y_train=sort_by_len(x_train,y_train)
x_text,y_test=sort_by_len(x_text,y_test)
def write_file(f_path,xs,ys):
    with open(f_path,'w',encoding='utf-8')as f:
        for x,y in zip(xs,ys):
            f.write(str(y)+'\t'+' '.join(idx2word[i]for i in x[1:])+'\n')
write_file('./data/train.txt',x_train,y_train)
write_file('.data/test.txt',x_text,y_test)
#构建语料表，基于词频进行统计
couter=Counter()
with open('./data/train.txt',encoding='utf-8')as f:
    for line in f:
        line=line.strip()
        label,words=line.split('\t')
        words=words.split(' ')
        couter.update(words)
words=['<pad>']+[w for w,freq in couter.most_common()if freq>=10]
print('Vocab size:',len(words))
Path('./vocab').mkdir(exist_ok=True)
with open('./vocab/word.text','w',encoding='utf-8')as f:
    for w in words:
        f.write(w+'\n')
#得到新的word2id映射表
word2idex={}
with open('./vocab/word.text',encoding='utf-8')as f:
    for i,line in enumerate(f):
        line=line.strip()
        word2idx[line]=i
#embedding层
'''
可以基于网络来训练，也可以直接加载别人训练好的，一般是加载预训练模型
'''
embedding=np.zeros(len(word2idx)+1,50)#+1表示如果不在语料表中，都是unkonw
with open('./data/glove.6B.50d.txt',encoding='utf-8')as f:#下载好的
    count=0
    for i,line in enumerate(f):
        if i%100000==0:
            print('-At line{}'.format(i))#打印处理了多少数据
        line=line.strip()
        sp=line.split(' ')
        word,vec=sp[0],sp[1:]
        if word in word2idx:
            count+=1
            embedding[word2idx[word]]=np.asanyarray(vec,dtype='float32')#词将转成对应的向量
#现在已经得到每个词表所对应的向量
print('[%d/%d] words have found pre_trained values'%(count,len(word2idx)))
np.save('./vocab/word.npy',embedding)
print('Saved ./vocab/word.npy')
#构建训练集
#注意所有的输入样本必须是相同的shape(文本的长度，词向量维度等)
#数据生成器
def data_generator(f_path,params):
    with open(f_path,encoding='utf-8')as f:
        for line in f:
            line=line.strip()
            label,text=line.split('\t')
            text=text.split(' ')
            x=[params['word2idx'].get(w,len(word2idx))for w in text]
            if len(x)>=params['max_len']:
                x=x[:params['max_len']]
            else:
                x+=[0]*(params['max_len'])-len(x)
            y=int(label)
            yield x,y
def dataset(is_training,params):
    _shapes=([params['max_len']],())
    _types=(tf.int32,tf.int32)
    if is_training:
        ds=tf.data.Dataset.from_generator(
            lambda :data_generator(params['train_path'],params),
            output_shapes=_shapes,
            output_types=_types,)
        ds=ds.shuffle(params['num_samples'])
        ds=ds.batch(params['batch_size'])
        ds=ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        ds=tf.data.Dataset.from_generator(
            lambda:data_generator(params['test_path'],params),
            output_shapes=_shapes,
            output_types=_types,)
        ds=ds.batch(params['batch_size'])
        ds=ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
'''
自定义网络模型
定义好都有那些层

'''
class Model(tf.keras.Model):
    def __init__(self,params):
        super().__init_()
        self.embedding=tf.Variable(np.load('./vocab/word.py'),
                                   dtype=tf.float32,
                                   name='pretrained_embedding',
                                   trainable=False)
        self.drop1=tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop2=tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop3=tf.keras.layers.Dropout(params['dropout_rate'])
        self.rnn1=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['run_uints'],return_sequences=True))
        self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['run_uints'], return_sequences=True))
        self.rnn3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['run_uints'], return_sequences=False))
        self.drop_fc=tf.keras.layers.Dropout(params['droupt_rate'])
        self.fc=tf.keras.layers.Dense(2*params['run_unints'],tf.nn.elu)
        self.out_linear=tf.keras.layers.Dense(2)

    def call(self,inputs,training=False):
        if inputs.dtype!=tf.int32:
            inputs=tf.cast(inputs,tf.int32)
        batch_sz=tf.shape(inputs)[0]
        run_unints=2*params['run_unints']
        x=tf.nn.embedding_lookup(self.embedding,inputs)
        x=self.drop1(x,training=training)
        x=self.rnn1(x)
        #x=tf.reshape(x,(batch_sz*10,10,run_unints))
        x=self.drop2(x,training=training)
        x=self.rnn2(x)
        #x=tf.reduce_max(x,1)
        #x=tf.reshape(x,(batch_sz))
        x=self.drop3(x,training=training)
        x=self.rnn3(x)
        x=self.drop_fc(x,training=training)
        x=self.fc(x)
        x=self.out_linear(x)
        return x

#设置参数
params={
    'vocab_path':'./vocab/word.txt',
    'train_path':'./data/train.txt',
    'test_path':'./data/test.txt',
    'num_samples':25000,
    'num_labels':2,
    'batch_size':32,
    'max_len':1000,
    'run_units':200,
    'dropout_rate':0.2,
    'clip_norm':10.,
    'num_patience':3,
    'lr':3e-4,
}
#用来判断进行提前停止
def is_descending(history:list):
    history=history[-(params['num_patience']+1):]
    for i in range(1,len(history)):
        if history[i-1]<=history[i]:
            return  False
    return True
word2idx={}
with open(params['vocab_path'],encoding='utf-8')as f:
    for i,line in enumerate(f):
        line=line.strip()
        word2idx[line]=i
params['word2idx']=word2idx
params['vocab_size']=len(word2idx)+1
model=Model(params)
model.build(input_shape=(None,None))
decay_lr=tf.optimizers.schedules.ExponentialDecay(params['lr'],1000,0.95)
optim=tf.optimizers.Adam(params['lr'])
global_step=0
history_acc=[]
best_acc=.0
t0=time.time()
logger=logging.getLogger('tensorflow')
logger.setLevel(logger.INFO)


while True:
    #训练模型
    for texts,labels in dataset(is_training=True,params=params):
        with tf.GradientTape()as tape:
            logits=model(texts,training=True)
            loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
            loss=tf.reduce_mean(loss)
        optim.lr.assign(decay_lr(global_step))
        grads=tape.gradient(loss,model.trainable_variables)
        grads,_=tf.clip_by_global_norm(grads,params['clip_norm'])
        optim.apply_gradients(zip(grads,model.trainable_variables))

        if global_step%50==0:
            logger.info('Step{}| Loss:{:.4f}|Spent:{:.1f} secs|LR:{:6F}'.format(global_step,loss.numpy().item(),time.time()-t0,optim.lr.numpy().item()))
            t0=time.time()
        global_step+=1
        #验证效果
        m=tf.keras.metrics.Accuracy()
        for texts,lables in dataset(is_training=False,params=params):
            logits=model(texts,trainng=False)
            y_pred=tf.argmax(logits,axis=1)
            m.update_state(y_true=labels,y_pred=y_pred)
        acc=m.result().numpy()
        logger.info('Evaluation:Testing Accuracy:{:.3f}'.format(acc))
        history_acc.append(acc)
        if acc>best_acc:
            best_acc=acc
        logger.info('Best Accuracy:{:.3f}'.format(best_acc))
        if len(history_acc)>params['num_patience']and is_descending(history_acc)
            logger.info('Testing Accuracy not improved over{}epochs,Early Stop'.format(params['num_patience']))
            break


