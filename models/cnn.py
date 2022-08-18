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

from modelOptions import get_cnn_args
from utils.data_loader import MyDataSet
class cnn(nn.Module):
    def __init__(self,args):
        super(cnn, self).__init__()
        self.dropout = 0.2
        W=args.words_dict
        D=args.embed_dim
        C = args.class_num
        n_filters=args.n_filters            #有多少个特征图
        Ks = args.kernel_sizes              #卷积核大小
        self.embedding=nn.Embedding(W,D)    
        """ cnn """
        # self.cnn=nn.Conv1d(in_channels=D,out_channels=n_filters,kernel_size=Ks,stride=1)
        # conv2d处理文本数据时，输入特征默认都为1，与con1d不同的是，kerner_size和stride会变成二维的
        self.conv=[nn.Conv2d(1,n_filters,(kk,D),stride=1) for kk in Ks]
        # for cnn cuda
        for conv in self.conv:
            conv = conv.cuda()
        # self.conv2=nn.ModuleList([nn.Conv2d(1,n_filters,(kk,D),stride=1) for kk in Ks])
        self.linear=nn.Linear(in_features=n_filters*len(Ks),out_features=C) #乘以1是因为卷积核的个数只有一个
    def forward(self,x):
        embed=self.embedding(x) # embed：(batch_size,seq_len,embed)  
        # cnn_input=self.dropout(x)
        cnn_input=embed.unsqueeze(1)  # (N,1,W,D) D:表示向量维度，W表示句子长度
        # 卷积-激活 (cnn的输出 当成bilstm的输入，在cnn层不需要池化)
        #  卷积后:(N,Co,W-kernersize+1/stride) * len(Ks)
        cnn_input = [F.relu(conv(cnn_input)).squeeze(3) for conv in self.conv] #[(N,Co,W), ...]*len(Ks)
        cnn_input = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_input] #[(N,Co), ...]*len(Ks)
        cnn_input = torch.cat(cnn_input, 1) # (N,len(Ks)*Co)
        # x = self.dropout(x)  
        logit = self.linear(cnn_input)
        return logit

def main():
    print("starting...")
    args=get_cnn_args()
    model=cnn(args)
    # x= torch.arange(3*32,dtype= torch.long).reshape(3,32)
    # 测试dataloader
    dataset=MyDataSet(args.file_path,args.vocab_path)
    dataloader=DataLoader(dataset,batch_size=3,shuffle=False, drop_last=True, num_workers=0)
    count=0
    for step,(x,y) in enumerate(dataloader):
        count+=1
        if count>5:
            break
        print(model(x))
        # print(model(x).shape)
    y_hat = model(x)
    print(y_hat.shape)

if __name__ == '__main__':
    main()