#!/usr/bin/env python3.8
# coding: utf-8

"""
gensim-lda 主题分析, pyecharts WordCloud 词云图
1.数据获取
2.数据合并
3.数据去重
4.评论内容合并: 一级评论内容+评论内容+评论表情

5.停用词导入, 扩充停用词
6.分词, 词筛选, 单字符词意义不大剔除

7.lda模型构建
8.lda模型训练, 生成话题df

9.词频统计
10.生成前xx个词的词云图 chart-w.html
"""

import pandas as pd
import numpy as np

import re
import jieba
import os
from gensim import corpora, models

import matplotlib.pyplot as plt
from pyecharts.charts import WordCloud


# 数据获取
df_l = []
for path_i in os.listdir("./艾滋病知识微博评论"):
    df_i = pd.read_excel("./艾滋病知识微博评论/" + path_i)
    df_l.append(df_i)

# 数据合并
df_all = pd.concat(df_l, ignore_index=True)
print("去重前数据长度：", len(df_all))

# 数据去重
df_all.drop_duplicates(inplace=True)
print("去重后数据长度：", len(df_all))


# 评论内容合并：一级评论内容+评论内容+评论表情
def combined_content(x,y,z): 
    if pd.isnull(x):
        x = ""
    else:
        x = x
    if pd.isnull(y):
        y = ""
    else:
        y = y
    if pd.isnull(z):
        z = ""
    else:
        z= z
    com = x+y+z    
    return com

  
df_all["combined"] = df_all[["一级评论内容", "评论内容", "评论表情"]].apply(lambda x: 
                                         combined_content(x[0], x[1], x[2]), axis=1)


# stopwords
with open("stopWords.txt", mode="r", encoding="utf8") as ff:
    stop_words = ff.readlines()
    stop_words = [stop_words_i.replace("\n", "") for stop_words_i in stop_words]
    stop_words.extend(["回复", "x000d", ""])


def content_process(content, stopwords):
    
    content = str(content).lower()
    pattern = re.compile("[<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+")
    content = re.sub(pattern, "", content)
    text = list(filter(lambda x: x not in stopwords, jieba.lcut(content)))
    
    return text

c_l = list(map(lambda x: content_process(x, stop_words), df_all["combined"]))
c_l = [list(filter(lambda x:len(x)>1, i)) for i in c_l]
c_l = list(filter(lambda x: len(x)>0, c_l))


# lda mode
def lda_train_topic(c_l):
    new_dict = corpora.Dictionary(c_l)
    corpus = [new_dict.doc2bow(cc) for cc in c_l]
    lda_model = models.LdaModel(corpus=corpus, num_topics=10, id2word=new_dict, passes=5)
    # lda.save("lda.model")
    
    topic_df = pd.DataFrame(columns=["topic_id", "words"])
    for i in range(9):
        topic_i = lda_model.print_topic(i, topn=8)
        x1 = topic_i.replace("\"", "")
        x1 = x1.split("+")
        topic_df.loc[i, "topic_id"] = i
        topic_df.loc[i, "words"] = "_".join([w.split("*")[-1].strip() for w in x1])
        
    return topic_df

topic_df = lda_train_topic(c_l)
topic_df

# 创建的lda模型，计算模型的困惑度，即模型质量，分数越低，质量越好
new_dict = corpora.Dictionary(c_l)
corpus = [new_dict.doc2bow(cc) for cc in c_l]
lda_model.log_perplexity(corpus)

# 查看数样本主题概率
lda_model.get_document_topics(new_dict.doc2bow(c_l[5]))

# 查看数样本主题概率
abc= lda_model.get_document_topics([new_dict.doc2bow(c_l[5])])
for i in abc:
    print(i)


# 词频
word_count_dict = {}
for text in c_l:
    for word in text:
        if word in word_count_dict:
            word_count_dict[word] +=1
        else:
            word_count_dict[word] = 1

word_count_dict_lim = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)[:150]

wc = WordCloud()
wc.add('WC', word_count_dict_lim, shape="circle", word_size_range=[15,45])
wc.render("chart-w.html")
