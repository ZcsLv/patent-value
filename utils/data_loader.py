
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
# from tqdm import tqdm
import re
from tqdm import tqdm
import numpy as np
import pickle,torch
# 构建数据集
import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#当前程序上一级目录
sys.path.append(BASE_DIR) #添加环境变量

UNK, PAD = '<UNK>', '<PAD>'
def load_dataset(file_path,vocab_path,pad_size=256):
    contents = []
    # 分词处理的匿名函数
    tokenizer = lambda x: [y for y in x]  # char-level
    with open(file_path,"rb") as f:
        datas=pickle.load(f)
        for i in range(len(datas)):
            lin = datas[i]
            content,label = lin['abstract'],lin['grade']
            # indictors=[lin['ind_claims'],lin['dep_claims'],lin['len_abstract'],
            # lin['len_claims'],lin['num_inventors'],lin['back_incitions'],lin['num_family'],lin['cpcs'],lin['ipcs']]
            " 有空的数据 就置为0"
            indictors=[int(lin['ind_claims']),int(lin['dep_claims']),int(lin['len_abstract']),
            int(lin['len_claims']),int(lin['num_inventors']),int(lin['back_incitions']),int(lin['num_family']),int(lin['cpcs']),int(lin['ipcs'])]
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id  从pkl文件中读出来保存好了的词表
            vocab=pickle.load(open(vocab_path,'rb'))
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append(( words_line,indictors,int(label), seq_len))
    return contents  # [([...], [],0,句子长度), ([...], 1), ...]

class MyDataSet(Dataset):
    # 一般用来写 数据获取,读取txt文件中的句子和label
    # 传入vocab_dict，训练、评估、测试共用一个词表。 
    def __init__(self,file_path,vocab_path) -> None:
        super().__init__()
        data=load_dataset(file_path,vocab_path,pad_size=256) # [([...], 0), ([...], 1), ...]
        self.examples=[]
        for i in data:
            self.examples.append(i)
    def __len__(self):
        return len(self.examples)
    " 通过复写 __getitem__() 方法 可通过索引 index 来访问数据，能够同时返回数据 和 标签label，这里的数据和标签都为 Tensor 类型。"
    def __getitem__(self,index):
        # data应该是转化为index后的
        x,y,z,_ = self.examples[index]
        x,y,z = torch.tensor(x),torch.tensor(y,dtype=torch.float32),torch.tensor(z)
        return x,y,z

# file_path,vocab_path='../data/indictors11_train.pkl','../data/vacab.pkl'
# dataset=MyDataSet(file_path,vocab_path)
# print(dataset[0])
# dataloader=DataLoader(dataset,batch_size=4,shuffle=False, drop_last=True, num_workers=0)
# # print(len(dataloader))
# count=0
# for train_iter, (x_train,indices_train,y_train) in enumerate(dataloader):
#     count+=1
#     print(train_iter)
# print(count) # count= len(dataloader) 7254/4=1813  所有的样本数/每次拿的样本数 = 