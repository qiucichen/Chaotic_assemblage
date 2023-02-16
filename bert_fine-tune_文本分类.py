#!/usr/bin/env python
# coding: utf-8

# bert_fine-tune,借助预训练模型进行文本分类
# 1-种子设置
# 2-数据加载处理
# 3-预训练模型加载，加载tokrnizer、model
# 4-文本tokenizer处理、dataset、dataloader
# 5-定义model
# 6-预训练model参数冻结、model实例化、优化器、损失函数-带权重loss
# 7-train和eval定义
# 8-train-fine tune
# 9-model加载进行predit

import numpy as np
import pandas as pd
import openpyxl
from datetime import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import transformers
from transformers import BertTokenizer, AutoModel
from transformers import AdamW

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score


random_seed = 2018
np.random.seed(random_seed)
torch.manual_seed(random_seed)               # 为cpu设置种子
torch.cuda.manual_seed(random_seed)          # 为gpu设置种子
torch.backends.cudnn.deterministic = True    # 算法确定，避免使用gpu并行不能复现


# 数据获取
df_d = pd.read_excel(r"./文本分类/data.xlsx")
print(df_d["label-A"].value_counts())

df_d.loc[df_d["label-A"]=="费用咨询", "label_B"] = 0
df_d.loc[df_d["label-A"]=="业务详情", "label_B"] = 1
df_d.loc[df_d["label-A"]=="网络问题", "label_B"] = 2
df_d.loc[df_d["label-A"]=="线下办理", "label_B"] = 3
df_d.loc[df_d["label-A"]=="办理不便", "label_B"] = 4

df_d.head(3)


# 数据计划分
train_text, test_text, train_label, test_label = train_test_split(df_d["text"], df_d["label_B"], 
                                                                 test_size=0.2, stratify = df_d["label_B"], random_state = random_seed)

# train_text.values[0]
# 文本长度分析
seq_len = [len(i) for i in train_text]
seq_df = pd.DataFrame(seq_len)
seq_df.describe()              # max_length----------------


# 加载 model
pretrained_model_path = "bert_base_chinese"

# 分词
tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
# print(tokenizer.batch_encode_plus(["下午好", "我爱北京天安门，你怎么样"],
#                             max_length=5, padding="max_length", truncation=True, return_tensors="pt"))

# model
model_bert = AutoModel.from_pretrained(pretrained_model_path)


max_length = 20
batch_size = 8

# 文本 tokenizer
token_train = tokenizer.batch_encode_plus(train_text.tolist(), max_length=max_length,
                                          padding="max_length", truncation=True, return_tensors="pt")
train_seq = token_train["input_ids"]
train_mask = token_train["attention_mask"]
train_y = torch.tensor(train_label.tolist(), dtype=torch.long)

token_test = tokenizer.batch_encode_plus(test_text.tolist(), max_length=max_length,
                                          padding="max_length", truncation=True, return_tensors="pt")
test_seq = token_test["input_ids"]
test_mask = token_test["attention_mask"]
test_y = torch.tensor(test_label.tolist(), dtype=torch.long)

# dataset
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, )

test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)


# model
class BertClassifier(nn.Module):
    def __init__(self, bert_pretrained_model, class_num):
        super(BertClassifier, self).__init__()
        self.class_num = class_num
        self.bert = bert_pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)       # bert的输出是768，转成512
        self.fc2 = nn.Linear(512, self.class_num)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, seq_ids, mask):
        bert_res = self.bert(seq_ids, attention_mask=mask)
        cls_hs = bert_res.pooler_output
        fc1_ = self.fc1(cls_hs)
        fc1_ = self.relu(fc1_)
        fc1_ = self.dropout(fc1_)
        fc2_ = self.fc2(fc1_)
        y = self.softmax(fc2_)
        return y


# 是否cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 学习率
lr = 2.6e-5

# 冻结部分层参数，加速训练（前几层不影响训练精度）
for param in model_bert.parameters():
    param.requires_grad = False
    
# model
model = BertClassifier(model_bert, 5)        # 与标签类别数量对应
model.to(device)
# 优化器
optimizer = AdamW(model.parameters(), lr)

# 样本均衡，关注数据少的样本,损失函数权重
print(df_d["label_B"].value_counts())
class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(train_label), y=train_label)

weights = torch.tensor(class_weight, dtype=torch.float)
weights.to(device)

# 损失函数
cross_entropy = nn.NLLLoss(weight=weights)


save_path = "bert_.model"
def train_epochs(epochs=2):
    best_valid_loss = float("inf")
    start_time = datetime.now()
    for epoch in range(epochs):
        print("== train {}/{}...".format(epoch+1, epochs))
        model.train()
        total_loss, total_accuracy, total_batch = 0, 0, 0
        total_preds, total_labels = [], []
        for step, batch in enumerate(train_dataloader):
            batch = [t.to(device) for t in batch]
            sent_ids, mask, labels = batch
            model.zero_grad()            # 梯度归零
            preds = model(sent_ids, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            loss.backward()             # 计算梯度，反向传播
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪，解决梯度爆炸，不解决梯度消失
            optimizer.step()            # 模型更新
            total_preds.append(preds.detach().cpu().numpy())
            total_labels.append(labels.detach().cpu().numpy())
            if step!=0 and step%50==0:
                current_time = datetime.now()
                cost_time = current_time-start_time
                print("epoch {}/{}, step {}, cost {}, loss {}".format(epoch+1, epochs, step, cost_time, loss))
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        avg_loss = total_loss/len(train_dataloader)
        preds = np.argmax(total_preds, axis=1)
        train_f1 = f1_score(total_labels, preds, average="micro")
        
        # 测试
        val_loss, val_f1, eval_preds, eval_labels = evaluate()
        if val_loss <= best_valid_loss:
            best_valid_loss = val_loss
            print("-->", val_loss)
#             torch.save(model.state_dict(), save_path)           # 保存模型方式-1-预测需要搭模型
            torch.save(model, save_path)                          # 保存模型方式-2-预测不需要搭模型
            
        current_time = datetime.now()
        print("Epoch {}/{}, train_loss:{}, train_f1:{} \n".format(epoch+1, epochs, avg_loss, train_f1), 
              "val_loss:{}, val_f1:{} \n".format(val_loss, val_f1),
              "best_valid_loss:{}, cost_time:{} \n".format(best_valid_loss, current_time-start_time),
              "{metrics_report}")
        total_batch +=1
    return model
    
def evaluate():
    print("eval...")
    model.eval()
    total_loss, total_accuracy= 0, 0
    total_preds, total_labels = [], []
    for step, batch in enumerate(test_dataloader):
        batch = [t.to(device) for t in batch]
        sent_ids, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_ids, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            total_preds.append(preds.detach().cpu().numpy())
            total_labels.append(labels.detach().cpu().numpy())
    avg_loss = total_loss/len(test_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    val_f1 = f1_score(total_labels, np.argmax(total_preds,axis=1), average="micro")
    
    return avg_loss, val_f1, total_preds, total_labels


# train
bert_fine_tune = train_epochs(6)

bert_fine_tune.eval()
aaa = tokenizer("我跟家人商量一下", return_tensors = "pt")
with torch.no_grad():
    preds = bert_fine_tune(aaa["input_ids"].to(device), aaa["attention_mask"].to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
preds






# ## predict


# predict
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer,AutoModel

tokenizer = BertTokenizer.from_pretrained("bert_base_chinese")
# model_bert = AutoModel.from_pretrained("bert_base_chinese")
# model
class BertClassifier(nn.Module):
    def __init__(self, bert_pretrained_model, class_num):
        super(BertClassifier, self).__init__()
        self.class_num = class_num
        self.bert = bert_pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)       # bert的输出是768，转成512
        self.fc2 = nn.Linear(512, self.class_num)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, seq_ids, mask):
        bert_res = self.bert(seq_ids, attention_mask=mask)
        cls_hs = bert_res.pooler_output
        fc1_ = self.fc1(cls_hs)
        fc1_ = self.relu(fc1_)
        fc1_ = self.dropout(fc1_)
        fc2_ = self.fc2(fc1_)
        y = self.softmax(fc2_)
        return y

device = torch.device("cpu")

# # model
# model___ = BertClassifier(model_bert, 5)
# model___.to(device)
# model___.load_state_dict(torch.load("bert_.model"))
# model___.eval()
model___ = torch.load("bert_.model")
model___.eval()

aaa = tokenizer("我跟家人商量一下", return_tensors = "pt")
with torch.no_grad():
    preds = model___(aaa["input_ids"].to(device), aaa["attention_mask"].to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
preds
