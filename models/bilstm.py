import torch 
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录
sys.path.append(BASE_DIR) #添加环境变量
from modelOptions import get_bilstm_args
from utils.data_loader import MyDataSet

class mylstm(nn.Module):
    def __init__(self,args) -> None:
        super(mylstm,self).__init__()
        self.num_layers=args.num_layers
        self.hidden_dim=args.lstm_hidden_dim
        self.classes=args.classes_nums
        W=args.words_dict
        D=args.embed_dim
        # W,D是使用的字典的长度,词向量维度。
        self.embed=nn.Embedding(W,D)
        #预训练词向量,输入特征应该是词向量维度
        self.bilstm=nn.LSTM(input_size=D,hidden_size=self.hidden_dim,num_layers=self.num_layers,bidirectional=True,dropout=0.2)
        
        self.fc=nn.Linear(self.hidden_dim*2,self.classes)
    def forward(self,x):
        # 传入的x是词典的index, 先经过embedding得到具体的词向量
        # x,_=x
        x=self.embed(x)         #embed前：（batch_size,seq_len)
                                #embed后：(batch_size,seq_len,word_dim)
        x=x.view(x.size(1),x.size(0),-1) # (seq_len,batch_size,word_dim)
        out,_=self.bilstm(x)        #self.bilstm的inputs：[input:（seq_len,batchsize,input_size),(h_0,c_0)]
                                #(h_0,c_0),h_0的size(num_layers * num_directions, batchsize, hidden_size)
                                # 如没有手动初始化，会自动初始化为0
                                #self.bilstm的outputs:[output, (h_n, c_n)]
                                #output:seq_len, batch, num_directions * hidden_size
                                #h_n:num_layers * num_directions, batch, hidden_size,c_n一样

        out=F.relu(out)         #seq_len, batch, num_directions * hidden_size
        out=torch.transpose(out,0,1)           # batch,seq_len, num_directions * hidden_size
        out=self.fc(out[:,-1,:])              #fc前：（batch_size,hiddend_size)
                                              #fc后：（batch_size,classes_nums)

        return out

# if __name__ == '__main__':
#     print("starting...")
#     args=get_bilstm_args()
#     print(args.num_layers)
#     model=mylstm(args)
#     # 测试dataloader
#     dataset=MyDataSet(args.file_path,args.vocab_path)
#     dataloader=DataLoader(dataset,batch_size=3,shuffle=False, drop_last=True, num_workers=0)
#     count=0
#     for step,(x,y) in enumerate(dataloader):
#         count+=1
#         if count>5:
#             break
#         print(model(x).shape)


# def main():