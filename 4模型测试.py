# -*- coding: utf-8 -*-
import jieba
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, losses, optimizers


lst_class = ['金融', '汽车', '食品', '房产', '科技', '健康']
BATCH_SIZE = 128  # batch 大小
TOTAL_WORDS = 20000  # 总词汇量
MAX_REVIEW_LEN = 500  # 评论最大词汇数量，不足句尾补零
EMBEDDING_LEN = 300  # 词向量维度
stopwords = [line.strip() for line in
             open('cn_stopwords.txt', encoding='utf-8').readlines()]
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


test_set = {
    '金融': "据新华社12月18日消息，中央财政在今年四季度增发1万亿元国债。日前，财政部已下达第一批资金预算2379亿元。专家认为，万亿国债在岁末年初将为基础设施建设带来增量资金。为巩固和增强经济回升向好态势，财政政策发力必要性持续提升。",
    '汽车': "11月15日，海马EX00（智行盒子）首批试制车下线仪式在郑州经开区海马汽车第三智能工厂成功举行，来自当地及相关部门的负责同志，智行盒子股东代表、合作伙伴代表共150余人共同见证海马EX00（智行盒子）首批试制车下线。",
    '食品': "说到牛奶蛋白过敏，很多家长并不陌生。由于婴幼儿肠道屏障发育不成熟、免疫系统发育不完善，极易发生牛奶蛋白过敏，并且已成为困扰很多婴幼儿和父母的一大健康难题。据不完全统计，中国婴儿牛奶蛋白过敏发生率约为2.5%—7%。受环境等因素影响，各地呈不同比例的上升趋势。",
    '房产': "市住建委昨天发布《关于做好本轮强降雪期间城镇房屋安全管理工作的通知》。通知明确，各区住建部门、房屋管理单位等要加大对城镇房屋巡查力度，重点对危房进行安全检查，同时要严密监控大跨度建筑顶部积雪并及时清除。",
    '科技': "光学晶体可实现频率转换、参量放大、信号调制等功能，是激光技术的“心脏”。经多年攻关，北京大学团队创造性提出新的光学晶体理论，并应用轻元素材料氮化硼首次制备出一种超薄、高能效的光学晶体“转角菱方氮化硼”（简称TBN），为新一代激光技术奠定理论和材料基础。",
    '健康': "12月16日，中国老年保健协会药品和器械合理使用及风险控制分会在北京成立，并召开了合理用药学术研讨会，国家相关部委负责人、中国药师协会会长张耀华等领导专家，以及全国26个临床医学、11个基础学科的200多名专家，参加了学术研讨，并见证了分会的成立。"
}
imported = tf.saved_model.load("./saver/news_recognition")
f = imported.signatures["serving_default"]
tplt = "{0:^4}\t{1:^20}\t{2:^20}\t{3:^20}\t{4:^20}\t{5:^20}\t{6:^20}"
for itm in test_set.items():
    lable = itm[0]
    para = np.array([[itm[1]]])
    x = list(map(preprocess, para))
    # 填充数组来保证输入数据具有相同的长度
    # keras只能接受长度相等的序列输入。使用pad_sequence()函数将序列转化为经过填充以后得到的一个长度相同新的序列。
    x = keras.preprocessing.sequence.pad_sequences(x, value=0,  # 填充0
                                                   padding='post',  # 填充在后面
                                                   maxlen=MAX_REVIEW_LEN)  # 最大长度
    x = tf.constant(x, dtype=tf.int32)
    result = f(x)['output_1'].numpy()
    print('======================================================================')
    print('新闻选段：' + para[0][0])
    print(tplt.format('分类:', '金融', '汽车', '食品', '房产', '科技', '健康'))
    print(tplt.format('概率:', *result[0]))
    print('【'+lable+'】'+"新闻的预测结果:"+'【'+lst_class[np.argmax(result)]+'】')
