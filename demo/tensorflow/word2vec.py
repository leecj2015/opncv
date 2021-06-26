import collections
import os
import random
import urllib
import zipfile
import numpy as np
import  tensorflow as tf
#train parameters
learning_rate=0.1
batch_size=128
num_steps=3000000
display_step=10000
eval_step=200000
#test sample
eval_words=['nine','of','going','hardware','american','britain']
#word2vec parameter
embeding_size=200#word2vec dimension
max_vocubulary_size=50000#语料库词语数量
min_occurence=10#Min vocubulary frequency
skip_window=3#左右窗口大小
num_skips=2#一次制作多少个输入和输出
num_sampled=64#负采样
#load trained data
data_path='C:/Users/50694/PycharmProjects/untitled1/demo/tensorflow/迁移学习/img/text8.zip'
with zipfile.ZipFile(data_path) as f:
    text_words=f.read(f.namelist()[0]).lower().split()
print(len(text_words))
#print(text_words)
#create one counter ,caculate vocubulary occur times
count=[('UNK',-1)]
#J基于词频返回max_vocabulary_size个常用词
count.extend(collections.Counter(text_words).most_common(max_vocubulary_size-1))
#print(count[0:10])
'''
min_occurence参数，判断每个次是否满足给定的条件
提出掉出现次数少于min_occurence的词语
'''
for i in range(len(count)-1,-1,-1):
    if count[i][1]<min_occurence:
        count.pop()
    else:
        break
'词-ID映射'
#计算语料库大小
vocabulary_size=len(count)
print('xxxxxxxx')
print(vocabulary_size)


#每个词分配一个ID
word2id=dict()
for i,(word,_) in enumerate(count):
    word2id[word]=i
    #print(word)
#print(word2id)
#所有词转换成ID
print('xxxxxxxxxxxxxxxxxxxxxxxx')
data=list()
unk_count=0
for word in text_words:
    #全部转换成ID
    index=word2id.get(word,0)
    if index==0:
        unk_count+=1
    data.append(index)
#print(data)
count[0]=('UNK',unk_count)
id2word=dict(zip(word2id.values(),word2id.keys()))
print('Words count:',len(text_words))
print('unique words:',len(text_words))
print('vocabulary_size:',vocabulary_size)
print('Most commont words',count[:10])
#构建所需训练数据
data_index=0
def next_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size%num_skips==0
    assert num_skips<=2*skip_window
    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    labels=np.ndarray(shape=(batch_size),dtype=np.int32)
    #get window size(words left and right_current one)
    span=2*skip_window+1#7维窗口，左右各3
    #创建一个长度维7的队列
    buffer=collections.deque(maxlen=span)
    if data_index+span>len(data):#如果数据被滑完一边
        data_index=0
    buffer.extend(data[data_index:data_index+span])
    data_index+=span
    for i in range(batch_size//num_skips):#num_skips表示去多少组不同的词作为输出，此例为2
        content_words=[w for w in range(span) if w!=skip_window]
        words_to_use=random.sample(content_words,num_skips)
        for j,content_word in enumerate(words_to_use):
            batch[i*num_skips+j]=buffer[skip_window]
            labels[i*num_skips+j,0]=buffer[content_word]
        if data_index==len(data):
            buffer.extend(data[0:span])
            data_index=span
        else:
            buffer.append(data[data_index])
            data_index+=1
    data_index=(data_index+len(data)-span)%len(data)
    return batch,labels
#ensure the following opts and var are assinged on cpu
#some opts not compatible on gpu
with tf.device('/cpu:0'):
    embeding=tf.Variable(tf.random.normal([vocabulary_size,embeding_size]))
    nec_weights=tf.Variable(tf.random.normal([vocabulary_size,embeding_size]))
    nec_biases=tf.Variable(tf.zeros([vocabulary_size]))

#通过tf.nn.embeding_lookup函数将索引转换成词向量
def get_embedding(x):
    with tf.device('/cup:0'):
        x_embed=tf.nn.embedding_lookup(embeding,x)
        return x_embed

#loss_function
'''
先分别计算正样本和负样本的对应的output and labels
then use sigmoid cross entropy caculate output and labels loss

'''
def nce_loss(x_embed,y):
    with tf.device('/cpu:0'):
        y=tf.cast(y,tf.int64)
        loss=tf.reduce_mean(
            tf.nn.nce_loss(weights=nec_weights,
                           biases=nec_biases,
                           labels=y,
                           inputs=x_embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))
        return loss
#evaluation
def evaluate(x_embed):
    with tf.device('/cup:0'):
        x_embed=tf.cast(x_embed,tf.float32)
        x_embed_norm=x_embed/tf.sqrt(tf.reduce_sum(tf.square(x_embed)))
        embeding_norm=embeding/tf.sqrt(tf.reduce_sum(tf.square(embeding),1,keepdims=True),tf.float32)
        cosine_sim_op=tf.matmul(x_embed_norm,embeding_norm,transpose_b=True)
        return cosine_sim_op
#SGD
optimizer=tf.optimizers.SGD(learning_rate)
#optimizer
def run_optimization(x,y):
    with tf.device('/cpu:0'):
        with tf.GradientTape() as g:
            emb=get_embedding(x)
            loss=nce_loss(emb,y)
            #caculate gradients
        gradients=g.gradient(loss,[embeding,nec_weights,nec_biases])
        #update
        optimizer.apply_gradients(zip(gradients,[embeding,nec_weights,nec_biases]))


#test
x_test=np.array(word2id[w.encode('utf-8')] for w in eval_words)
#train
for step in range(1,num_steps+1):
    batch_x,batch_y=next_batch(batch_size,num_skips,skip_window)
    run_optimization(batch_x,batch_y)
    if step%display_step==0 or step==1:
        loss=nce_loss(get_embedding(batch_x),batch_y)
        print('step:%i,loss:%f'%(step,loss))
    #evaluation
    if step%eval_step==0 or step==1:
        print('evaluation..........')
        sim=evaluate(get_embedding(x_test)).numpy()
        for i in range(len(eval_words)):
            top_k=8
            nearest=(-sim[i,:]).argsort()[1:top_k+1]
            log_str='"s%" nearest neighbors:'%eval_words[i]
            for k in range(top_k):
                log_str='%s %s'%(log_str,id2word[nearest[k]])
            print(log_str)










































































