# -*- coding: utf-8 -*-
import datetime
import jieba
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, losses, optimizers

BATCH_SIZE = 128  # batch_size：表示单次传递给程序用以训练的数据（样本）个数
TOTAL_WORDS = 20000  # 总词汇量
MAX_REVIEW_LEN = 500  # 评论最大词汇数量，不足句尾补零
EMBEDDING_LEN = 300  # 词向量维度
stopwords = [line.strip() for line in open('cn_stopwords.txt', encoding='utf-8').readlines()]
word_index = pickle.load(open('word_index.pkl', 'rb'))
word_embeddings = pickle.load(open('word_embeddings.pkl', 'rb'))
embedding_matrix = pickle.load(open('embedding_matrix.pkl', 'rb'))


def remove_stopwords(str):
    sen = [i for i in str if i not in stopwords]
    return sen


def preprocess(x):
    processed_x = []
    x = jieba.cut(x[0])
    x = remove_stopwords(x)
    for i in x:
        index = word_index.get(i, 0)
        if index < TOTAL_WORDS:
            processed_x.append(index)
        else:
            processed_x.append(0)
    return processed_x


def code_mark(y, c):
    i = 0
    class_num = c
    code = [j for j in c * [0.]]
    for item in y:
        code[item - 1] = 1.
        y[i] = code
        i += 1
        code = [j for j in c * [0.]]
    return y


data = pd.read_csv('data/all_news_text.csv')
data = data.dropna(axis=0, how='any')
x = np.expand_dims(data['news_text'].to_list(), axis=1)
y = data['mark'].to_list()
y = code_mark(y, 6)
# y = np.expand_dims(y, axis=1)
x = list(map(preprocess, x))
# 填充数组来保证输入数据具有相同的长度
# keras只能接受长度相等的序列输入。使用pad_sequence()函数将序列转化为经过填充以后得到的一个长度相同新的序列。
x = keras.preprocessing.sequence.pad_sequences(x, value=0,  # 填充0
                                               padding='post',  # 填充在后面
                                               maxlen=MAX_REVIEW_LEN)  # 最大长度
# x:(4497*80);y:(4497*1)
# x = np.expand_dims(x, axis=1)
train_db = tf.data.Dataset.from_tensor_slices((x, y))  # 每MAX_REVIEW_LEN个特征对应1个标签,组成一组(MAX_REVIEW_LEN,1)
train_db = train_db.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
# 先打乱buffer_size缓冲器10000，然后128‘batch_size’个分一组(批次大小)
# for index,line in enumerate(train_db):
#     print(index," ",line)
#     print('----------------------------')
class MyLSTM(keras.Model):
    def __init__(self, units):
        super(MyLSTM, self).__init__()
        self.embedding = layers.Embedding(TOTAL_WORDS, EMBEDDING_LEN,
                                          input_length=MAX_REVIEW_LEN,
                                          trainable=False)
        # 因为使用了预测练的词向量，这里trainable设置为False，当然也可以设置为True
        self.embedding.build(input_shape=(None, MAX_REVIEW_LEN))
        self.embedding.set_weights([embedding_matrix])
        # 传入预训练的词向量矩阵
        # 单向LSTM网络
        # self.rnn = Sequential([layers.LSTM(units, dropout=0.2, return_sequences=True),
        #                        layers.LSTM(units, dropout=0.2, return_sequences=False)])
        # 双向LSTM网络(BiLSTM)
        self.rnn = Sequential([layers.Bidirectional(layers.LSTM(units, dropout=0.2, return_sequences=True)),
                               layers.Bidirectional(layers.LSTM(units, dropout=0.2, return_sequences=False))])
        self.out_layer = Sequential([layers.Dense(32),
                                     layers.Dropout(rate=0.2),
                                     layers.ReLU(),
                                     layers.Dense(6)])
        # 即便你只想用一个全连接层输出，也必须用Sequential包装一下，
        # 用Sequential包装后才可接受training参数

    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        out = self.rnn(x)
        x = self.out_layer(out, training)
        x = tf.sigmoid(x)  # 0-1可以被表示做概率或者用于输入的归一化，平滑的渐变，防止输出值“跳跃”
        return x


model = MyLSTM(64)
model.compile(optimizer=optimizers.Adam(1e-3),
              loss=losses.BinaryCrossentropy(),  # 二元交叉熵
              metrics=['accuracy'],
              experimental_run_tf_function=False)
# log
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# tensorboard --logdir logs/fit
model.fit(train_db, epochs=50, callbacks=[tensorboard_callback])
tf.saved_model.save(model, './saver/news_recognition')
# model.save('news_recognition.h5')
