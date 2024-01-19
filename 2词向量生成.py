# -*- coding: utf-8 -*-
import jieba
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, losses, optimizers

TOTAL_WORDS = 20000  # 总词汇量
EMBEDDING_LEN = 300  # 词向量维度

data = pd.read_csv('data/all_news_text.csv')
data = data.dropna(axis=0, how='any')
x = np.expand_dims(data['news_text'].to_list(), axis=1)


def get_word_embeddings():
    word_embeddings = {}
    with open('sgns.sogou.char', encoding='utf-8') as f:
        lines = f.readlines()
        for _, line in enumerate(lines):
            # 不要第一行，第一行是词向量表的统计信息
            if _ != 0:
                values = line.split()
                # 每一行的数据按空格分开，就变成了词和词向量
                word = values[0]
                # 取词
                coefs = np.asarray(values[1:], dtype='float32')
                # 取词向量。参与计算的 tensor 一般 float32 即可满足计算精度，有需求可以自己调节
                word_embeddings[word] = coefs
    return word_embeddings


def remove_stopwords(str):
    sen = [i for i in str if i not in stopwords]
    return sen


def get_word_index():
    words = []
    # 储存所有的词
    cutted_x = [jieba.cut(i[0]) for i in x]

    cutted_clean_x = [remove_stopwords(i) for i in cutted_x]
    for i in cutted_clean_x:
        words.extend(i)
    word_index = pd.DataFrame(pd.Series(words).value_counts())
    # 生成一个 pdFrame，行名为词，第一列为该词出现次数，并按照出现次数排序

    print("word_index", word_index)
    word_index['id'] = list(range(1, len(word_index) + 1))
    print("word_index", word_index)
    # 新生成一列，对应每个词的数字编码 id
    word_index = word_index.drop('count', axis=1)
    print("word_index", word_index)
    # 删除表示出现次数的那一列
    word_index = word_index.to_dict()['id']
    # to_dict 方法转换后，是一个嵌套 dict
    # 词-编码表生成完毕
    return word_index


# 保存到本地
def get_embedding_matrix():
    num_words = min(TOTAL_WORDS, len(word_index))
    print("num", num_words)
    embedding_matrix = np.zeros((num_words, EMBEDDING_LEN))
    print("aaaa", embedding_matrix)
    for word, i in word_index.items():
        if i >= TOTAL_WORDS:
            continue
            # 过滤掉超出词汇
        embedding_vector = word_embeddings.get(word)
        # 查询词向量
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


stopwords = [line.strip() for line in open('cn_stopwords.txt', encoding='utf-8').readlines()]
word_embeddings = get_word_embeddings()
print(word_embeddings)
pickle.dump(word_embeddings, open('word_embeddings.pkl', 'wb'))
# #
# word_embeddings = pickle.load(open('word_embeddings.pkl', 'rb'))
# print(word_embeddings)
word_index = get_word_index()
pickle.dump(word_index, open('word_index.pkl', 'wb'))
##
embedding_matrix = get_embedding_matrix()
pickle.dump(embedding_matrix, open('embedding_matrix.pkl', 'wb'))
# """金融,汽车,食品,房产,科技,健康"""
