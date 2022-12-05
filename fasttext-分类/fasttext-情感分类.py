#!/usr/bin/env python
# coding: utf-8

import jieba
import pandas as pd
import os
import random
from sklearn.metrics import f1_score
import re
from tqdm import tqdm
import fasttext


def delete_tag(s):
    # 特殊字符处理
    s = re.sub('\{IMG:.?.?.?\}', '', s)                    #图片
    s = re.sub(re.compile(r'[a-zA-Z]+://[^\s]+'), '', s)   #网址
    s = re.sub(re.compile('<.*?>'), '', s)                 #网页标签
    s = re.sub(re.compile('&[a-zA-Z]+;?'), ' ', s)         #网页标签
    s = re.sub(re.compile('[a-zA-Z0-9]*[./]+[a-zA-Z0-9./]+[a-zA-Z0-9./]*'), ' ', s)
    s = re.sub("\?{2,}", "", s)
    s = re.sub("\r", "", s)
    s = re.sub("\n", ",", s)
    s = re.sub("\t", ",", s)
    s = re.sub("（", ",", s)
    s = re.sub("）", ",", s)
    s = re.sub("\u3000", "", s)
    s = re.sub(" ", "", s)
    r4 = re.compile('\d{4}[-/]\d{2}[-/]\d{2}')             #日期
    s = re.sub(r4,'某时',s)
    return s



def tokenizer(text):
    # 文本处理分词
    text = [ 
            list(filter(lambda x:x not in stop_words,jieba.lcut(re.sub(su,"",delete_tag(document))))) 
            for document in tqdm(text)
    ]
    return text

su = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？°é÷σ―′ⅰⅱ←↓√∶≤≥⊙─━│┏┛▁▌▍▎□△▼▽◆◇○◎●☆〇「〞の㎡“”‘’！[\\]^_`{|}~\s]+'


if __name__=="__main__":
    with open(r"stopWords.txt",mode="r",encoding="utf8") as ff:
        stop_words = ff.readlines()
        stop_words = [stop_words_i.replace("\n","") for stop_words_i in stop_words]

    # 数据获取与处理
    data_neg = pd.read_csv(r"./data/neg_1.csv")
    data_pos = pd.read_csv(r"./data/pos_1.csv")

    data_neg.drop_duplicates(inplace=True)
    data_pos.drop_duplicates(inplace=True)

    neg_list = data_neg["正文"].tolist()[:1500]
    pos_list = data_pos["正文"].tolist()[:1500]

    # 数据处理
    pos_list2 = tokenizer(pos_list)
    neg_list2 = tokenizer(neg_list)
    
    # 处理为满足fasttext数据格式
    neg_list3 = ["__label__" + str(0)+" "+" ".join(i) for i in neg_list2]
    pos_list3 = ["__label__" + str(1)+" "+" ".join(i) for i in pos_list2]
    # 训练集测试集
    train_neg_pos_list = neg_list3[0:1000]+pos_list3[0:1000]
    test_neg_pos_list = neg_list3[1000:1500]+pos_list3[1000:1500]

    random.seed(123)
    random.shuffle(train_neg_pos_list)

    # 模型相关
    print("writing data to fasttext format...")
    out = open("./data/train_data.txt", 'w', encoding='utf-8')
    for sentence in train_neg_pos_list:
        if len(sentence)>11:
            out.write(sentence+"\n")
    print("done!")

    # test 数据保存
    print("writing data to fasttext format...")
    out = open("./data/test_data.txt", 'w', encoding='utf-8')
    for sentence in test_neg_pos_list:
        if len(sentence)>11:
            out.write(sentence+"\n")
    print("done!")

    # 基本模型
    classifier = fasttext.train_supervised(input=".\\data\\train_data.txt",
                                           dim=128,
                                           epoch=50,
                                           lr=0.1, 
                                           wordNgrams=5, 
                                           loss= 'softmax',
                                          )
    # 模型保存
    # classifier.save_model('./model/classifier.model')

    # 测试集-测试
    print(classifier.test(".\\data\\test_data.txt"))

