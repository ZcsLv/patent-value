
import os,pickle
from tqdm import tqdm
import pickle as pkl
MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
import sys,os
# '/root/pro/learn-models-pytorch/models-script/__file__'
# BASE_DIR=os.path.abspath('__file__') 
# BASE_DIR = os.path.dirname(os.path.abspath('__file__')) #当前程序上一级目录，这里为mycompany
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__'))) #当前程序上上一级目录，这里为mycompany
# print(BASE_DIR)
sys.path.append(BASE_DIR) #添加环境变量

def create_vocab(data,ues_word,min_freq,max_size):
    # 用于移除字符串头尾指定的字符 默认为空行或者空格
    # texts_split = [' '.join(js['title'] + js['abstract']) for js in data]
    vocab_dic = {}
    # 分词的匿名函数
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    # 构建词表,词与index一一对应
    for word in tokenizer(data):
        for char in tokenizer(word):
            vocab_dic[char] = vocab_dic.get(char, 0) + 1        # 统计词出现的频率
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

def save_dataset_vocab_size(data,vocab_path):
    # pkl.dump(vocab, open(config.vocab_path, 'wb'))
    if os.path.exists(vocab_path):
        vocab = pkl.load(open(vocab_path, 'rb'))
    else:
        vocab = create_vocab(data,False,max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(vocab_path, 'wb'))
    # print(f"Vocab size: {len(vocab)}")
    return len(vocab)

"""获取train val的文本内容构造词表"""
def get_content_data(train_file_path, valid_file_path):
    data = []
    # 读取pkl 文件中的 摘要 文本内容
    with open(train_file_path,"rb") as f:
        datas=pickle.load(f)
        for i in range(len(datas)):
            lin = datas[i]
            content= lin['abstract']
            data.append(content)
    with open(valid_file_path,"rb") as f:
        datas=pickle.load(f)
        for i in range(len(datas)):
            lin = datas[i]
            content= lin['abstract']
            data.append(content)
    return data

vocab_path,train_file_path,valid_file_path="data/vacab.pkl","data/indictors11_train.pkl","data/indictors11_test.pkl"
content=get_content_data(train_file_path,valid_file_path)
# print(content)
vocab_dic=create_vocab(content,False,min_freq=1,max_size=MAX_VOCAB_SIZE)
# print(vocab_dic)
# print(len(vocab_dic))
ll=save_dataset_vocab_size(content,vocab_path)
