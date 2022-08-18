import torch 
import torch.nn as nn
import torch.nn.functional as F
import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录
sys.path.append(BASE_DIR) #添加环境变量

from utils.data_loader import MyDataSet
device=torch.device("cuda:2")
""" Early detection of valuable patents using a deep learning model Case of semiconductor industry论文复现
    (cnn 并联 lstm ) 并联 (cnn 并联 lstm ) 并联 多层感知机 """

" 用于构建 abstract claim 特征的模型 "
class Sub_cnn_bilstm(nn.Module):
    def __init__(self,args) -> None:
        super(Sub_cnn_bilstm, self).__init__()

        self.dropout = 0.2
        W=args.words_dict
        D=args.embed_dim
        Ci=1
        self.n_filters=args.n_filters            # 有多少个特征图
        Ks = args.kernel_sizes              # 卷积核大小

        " 预训练词向量 "
        if args.word_embedding:
            self.embedding.weight.data.copy_(args.pretrained_weight)
        else:
            self.embedding=nn.Embedding(W,D)    
        " cnn "
        self.conv=[nn.Conv2d(1,args.n_filters,(kk,D),stride=1) for kk in Ks]
         # for cnn cuda
        for conv in self.conv:
            conv = conv.cuda()
        # self.conv=nn.ModuleList([nn.Conv2d(Ci, n_filters, (K, D), stride=1, padding=(K//2, 0)) for K in Ks])

        " bilstm "
        # 串联：n_filters是con层的输出通道数，作为lstm的输入通道数
        # 串联：词向量维度 作为lstm的输入通道数
        self.bilstm = nn.LSTM(D, args.lstm_hidden_dim, num_layers=args.lstm_num_layers, bidirectional=True)

        " dropout "
        self.dropout=nn.Dropout(self.dropout)

    def forward(self,x):
        # cnn 
        embed=self.embedding(x)                                     # (B,S,E)
        cnn_x=embed.unsqueeze(1)                                    # (B,1,S,E)
        cnn_x=[F.relu(conv(cnn_x)).squeeze(3) for conv in self.conv]           # (B,Co,S-ks+1/stride) * len(ks)
        cnn_x=[F.max_pool1d(i,i.size(2)).squeeze(2) for i in cnn_x] # (B,Co) * len(ks)
        cnn_y=torch.cat(cnn_x,1)                                    # (B,Co * len(ks))
        cnn_y= cnn_y.view(cnn_y.size(0), self.n_filters,-1)    # (B,Co,len(Ks))      # (B,Co,len(Ks))

        # bilstm
        bilstm_x=embed.view(embed.size(1),embed.size(0),-1)     # (S,B,E)
        bilstm_y,_=self.bilstm(bilstm_x)                        # (S,B,num_directions * hidden_size)
        bilstm_y=F.relu(bilstm_y)                               # (S,B,num_directions * hidden_size)
        bilstm_y=torch.transpose(bilstm_y,0,1)                  # (B,S,num_directions * hidden_size)
        bilstm_y=(bilstm_y[:,-1,:])                             # (B,num_directions * hidden_size) num_directions * hidden_size要求=n_filters
        bilstm_y=bilstm_y.unsqueeze(2)                          # (B,num_directions * hidden_size,1) num_directions * hidden_size要求=n_filters
       
        out=torch.cat((cnn_y,bilstm_y),2)                       # (B,Co,len(ks)+1)  (B,K,4)

        return out

" 用于构建 indices 特征的模型 "
class Indices(nn.Module):
    def __init__(self,args) -> None:
        super(Indices,self).__init__()
        Q=args.indexs
        K=args.n_filters
        self.fc=nn.Linear(Q,4*K)
    def forward(self,x):
        y=self.fc(x)                # (B,Q)->(B,4*K)
        y=y.view(y.size(0),-1,4)    # (B,4*K)->(B,K,4)
        return y

" 并联abstract claims fc"
class mcbilstm(nn.Module):
    def __init__(self,args) -> None:
        super(mcbilstm,self).__init__()
        " 用于特征混合的参数 "
        self.w_abs=nn.Parameter(torch.randn(args.batch_size,args.n_filters,4))
        self.w_claims=nn.Parameter(torch.randn(args.batch_size,args.n_filters,4))
        self.w_indices=nn.Parameter(torch.randn(args.batch_size,args.n_filters,4))

        self.abs=Sub_cnn_bilstm(args)
        # self.cls=Sub_cnn_bilstm(args)
        self.ins=Indices(args)
        " 两个fc "
        self.fc1=nn.Linear(args.n_filters*4,args.n_filters*2)
        self.fc2=nn.Linear(args.n_filters*2,args.class_num)
        
    def forward(self,abstract,indices):
        abs=self.abs(abstract)          # (B,K,4)
        # claims=self.cls(claims)         # (B,K,4)
        indices=self.ins(indices)       # (B,K,4)
        " feature fusion "
        out=self.w_abs* abs+self.w_indices* indices  # (B,K=256,4)
        out=out.view(out.size(0),1,-1)  # (B,1,K*4) 
        out=out.squeeze()            # (B,K*4)
        out=self.fc1(out)               # (B,K*2)
        out=self.fc2(out)               # (B,classes_nums)
        return out