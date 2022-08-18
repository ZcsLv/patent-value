from ast import arg
import torch 
import torch.nn as nn
import torch.nn.functional as F
# 参考：https://www.jianshu.com/p/45a26d278473
# 参考(point)：https://blog.csdn.net/sunny_xsc1994/article/details/82969867
# 
from torch.utils.data.dataloader import DataLoader
import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录
sys.path.append(BASE_DIR) #添加环境变量

from modelOptions import get_cnn_ind_args
from utils.data_loader import MyDataSet

" 用于构建 处理摘要的cnn结构"
class cnn(nn.Module):
    def __init__(self,args):
        super(cnn, self).__init__()
        self.dropout = 0.2
        W=args.words_dict
        D=args.embed_dim
        C = args.class_num
        self.n_filters=args.n_filters       #有多少个特征图
        Ks = args.kernel_sizes              #卷积核大小
        self.embedding=nn.Embedding(W,D)    
        """ cnn """
        # self.cnn=nn.Conv1d(in_channels=D,out_channels=n_filters,kernel_size=Ks,stride=1)
        # conv2d处理文本数据时，输入特征默认都为1，与con1d不同的是，kerner_size和stride会变成二维的
        self.conv=[nn.Conv2d(1,self.n_filters,(kk,D),stride=1) for kk in Ks]
        # for cnn cuda
        for conv in self.conv:
            conv = conv.cuda()
        # self.conv2=nn.ModuleList([nn.Conv2d(1,self.n_filters,(kk,D),stride=1) for kk in Ks])

        self.linear=nn.Linear(in_features=self.n_filters*len(Ks),out_features=C) #乘以1是因为卷积核的个数只有一个
    def forward(self,x):
        embed=self.embedding(x) # embed：(batch_size,seq_len,embed)  
        " avier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out) "
        # cnn_input=self.dropout(x)
        cnn_input=embed.unsqueeze(1)  # (N,1,W,D) D:表示向量维度，W表示句子长度
        # 卷积-激活 (cnn的输出 当成bilstm的输入，在cnn层不需要池化)
        #  卷积后:(N,Co,W-kernersize+1/stride) * len(Ks)
        cnn_input = [F.relu(conv(cnn_input)).squeeze(3) for conv in self.conv] #[(N,Co,W), ...]*len(Ks)
        cnn_input = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_input] #[(N,Co), ...]*len(Ks)
        cnn_input = torch.cat(cnn_input, 1) # (N,len(Ks)*Co)
        cnn_input = cnn_input.view(cnn_input.size(0), self.n_filters,-1) # (N,Co,len(Ks))
        # x = self.dropout(x)  
        # logit = self.fc(x)
        return cnn_input

" 用于构建 indices 特征的模型 "
class Indices(nn.Module):
    def __init__(self,args) -> None:
        super(Indices,self).__init__()
        Q=args.indexs
        K=args.n_filters
        self.fc=nn.Linear(Q,3*K)    # 3 表示卷积核的个数
    def forward(self,x):
        y=self.fc(x)                # (B,Q)->(B,4*K)
        y=y.view(y.size(0),-1,3)    # (B,4*K)->(B,K,4)
        return y

" 用于构建cnn + indices 的并联模型"
class union(nn.Module):
    def __init__(self,args) -> None:
        super(union,self).__init__()
        " 用于特征混合的参数 "
        self.w_abs=nn.Parameter(torch.randn(args.batch_size,args.n_filters,3))
        self.w_ins=nn.Parameter(torch.randn(args.batch_size,args.n_filters,3))
        self.abs=cnn(args)
        self.ins=Indices(args)
        self.fc1=nn.Linear(len(args.kernel_sizes)*args.n_filters,args.class_num)
    def forward(self,abstract,indices):
        abs=self.abs(abstract)          # (B,K,4)
        indices=self.ins(indices)       # (B,K,4)
        " feature fusion "
        out=self.w_abs* abs+self.w_ins* indices
        out=out.view(out.size(0),1,-1)  # (B,1,K*4)
        out=out.squeeze()            # (B,K*4)
        out=self.fc1(out)               # (B,classes_nums)
        return out
def main():
    print("starting...")
    args=get_cnn_ind_args()
    model=cnn(args)
    model2=Indices(args)
    model3=union(args)
    # x= torch.arange(3*32,dtype= torch.long).reshape(3,32)
    # 测试dataloader
    # dataset=MyDataSet(args.file_path,args.vocab_path)
    dataset=MyDataSet("../data/indictors11_train.pkl","../data/vacab.pkl")
    dataloader=DataLoader(dataset,batch_size=4,shuffle=False, drop_last=True, num_workers=0)
    count=0
    for step,(x1,x2,y) in enumerate(dataloader):
        count+=1
        if count>5:
            break
        # print(model(x1))
        # print(model(x2))
        # print(model(x).shape)
    y_hat = model(x1)
    y_hat2 = model2(x2)
    y=model3(x1,x2)
    print(y_hat.shape)
    print(y_hat2.shape)
    print(y.shape)

if __name__ == '__main__':
    main()