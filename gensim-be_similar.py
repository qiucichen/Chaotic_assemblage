# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-

import jieba
from gensim import corpora, models, similarities


class SimHunt(object):
    def __init__(self, contents_words_list):
        self.contents_words_list = contents_words_list
        self.dictionary, self.tfidf_model, self.index_x, _, _ = self.corpora_train()

    def corpora_train(self):
        dictionary = corpora.Dictionary(self.contents_words_list)
        dictionary.compactify()
        bow = list(map(lambda x: dictionary.doc2bow(x, allow_update=True), self.contents_words_list))
        tfidf_model = models.TfidfModel(bow)
        tfidf_vec = list(map(lambda x: tfidf_model[x], bow))
        index_x = similarities.Similarity("E:\\tmp", tfidf_vec, len(dictionary))

        return dictionary, tfidf_model, index_x, bow, tfidf_vec

    def topk(self, list_t, k=3):
        temporary = list_t.copy()

        max_number = []
        for _ in range(k):
            number_one = max(temporary)
            index = temporary.index(number_one)
            temporary[index] = 0
            max_number.append((index, number_one))
        return max_number

    def similar_index(self, single_content, k=3):
        single_bow = self.dictionary.doc2bow(single_content)
        new_tfidf = self.tfidf_model[single_bow]
        new_pro = self.index_x[new_tfidf]
        index_list = self.topk(new_pro.tolist(), k)
        return index_list


if __name__ == "__main__":
    content_list = ["十分钟视频看中美元首巴厘岛会晤", "G20“重头戏”开启 中美元首巴厘岛会晤",
                    ",中国国家主席习近平在印度尼西亚巴厘岛同美国总统拜登举行会晤", "这次会晤中的多个细节向世界传递出复杂但重要的信号"]
    cut_content_list = [jieba.lcut(content) for content in content_list]
    sim_ = SimHunt(cut_content_list)

    pre_ = jieba.lcut("从四个细节,看中美元首巴厘岛会晤")
    pre = sim_.similar_index(pre_, k=3)
