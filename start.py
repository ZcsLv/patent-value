
from train_eval_test import train,test
from torch.utils.data.dataloader import DataLoader
from modelOptions import get_bilstm_args,get_cnn_args,get_cnn_ind_args,get_cnnbilstm_inds_args
from options import get_train_args

from models.cnn import cnn
from models.cnn_inds import union
from models.cnnbilstm_inds import mcbilstm

from utils.data_loader import MyDataSet



if __name__ == '__main__':

    args_train=get_train_args()
    args_model=get_cnnbilstm_inds_args()
    # " model name "
    # args_train.add_argument('--model_name', type=str, default="mcbilstm")

    train_dataset=MyDataSet(args_train.train_file_path,args_train.vocab_path)
    dev_dataset=MyDataSet(args_train.dev_file_path,args_train.vocab_path)
    test_dataset=MyDataSet(args_train.test_file_path,args_train.vocab_path)

    train_dataloader=DataLoader(train_dataset,batch_size=args_train.batch_size,shuffle=True, drop_last=True, num_workers=2)
    dev_dataloader=DataLoader(dev_dataset,batch_size=args_train.batch_size,shuffle=True, drop_last=True, num_workers=2)
    test_dataloader=DataLoader(test_dataset,batch_size=args_train.batch_size,shuffle=True, drop_last=True, num_workers=2)

    train(train_dataloader,dev_dataloader,args_train,mcbilstm(args_model))