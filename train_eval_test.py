import torch
import torch.nn as nn
import logging,time
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,precision_score,f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.logger import create_logger


" loggin 文件日志 "
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_logger =create_logger('log/cnnlstm-inds-train-v1.log')
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.INFO)
# handler = logging.FileHandler('log/cnn-train.log', 'w', 'utf-8')
# root_logger.addHandler(handler)
def standar(data):
    st = StandardScaler()
    data = st.fit_transform(data)  # 同一列的为一个特征
    return data
def train(train_iter,dev_iter,args,model):
    logger=SummaryWriter(log_dir="ts/"+args.model_name)
    start_time = time.time()
    logging.info("Training start ......")
    print("Training start time:",start_time)
    #  模型、数据放到gpu上。 方法1）：.cuda()   方法2）：.to(device) device是指定好的
    model.to(device)
    # step 1:定义优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    # 2) 定义损失函数
    lossf=nn.CrossEntropyLoss()
    dev_best_loss = float('inf')
    last_improve,iter=0,0
    # step 2:
    for epoch in range(args.epoches):
        root_logger.info("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epoches))
        # step 3: 读取batch个数据
        # print("len(train_iter):",len(train_iter))
        for steps,(x1,x2,y) in enumerate(train_iter):
                # 指定模型的  方式
            model.train()
            # 1) 准备数据
            # 标准化处理
            x2=torch.tensor(standar(x2),dtype=torch.float32)
            text,inds,label=x1.to(device),x2.to(device),y.to(device)
            # 3）梯度归零
            optimizer.zero_grad()
            # 4) 
            logit=model(text,inds)
            # 5) 计算loss
            loss=lossf(logit,label)  
            # 6) 
            loss.backward() 
            optimizer.step()
            iter+=1
            if steps % args.log_intervel==0:
                # corrects = (torch.max(logit, 1)[1].view(label.size()).data == label.data).sum()
                # accuracy = float(corrects)/args.batch_size * 100.0
                " 另一种写法 "
                y_pred=torch.max(logit,1)[1]     # 共10类，0-9类，会得到 [2,1,3,4,5,...],[1]表示最大值索引,从0开始
                " 计算结果时，需要将数据转到cpu上"
                accuracy=accuracy_score(label.data.cpu().numpy(),y_pred.cpu().numpy())
                confusion=confusion_matrix(label.data.cpu().numpy(),y_pred.cpu().numpy())
                # 是用loss 还是用loss.item?
                root_logger.info("\n steps: {}   ,train_loss: {} , train_acc, {}".format(iter,loss.item(), accuracy)) # loss还是一个tensor变量，要用item得到值
                # 添加日志
                logger.add_scalar("train_loss", loss.item(), global_step=steps)
                logger.add_scalar("train_accuracy", accuracy, global_step=steps)
            if steps % args.eval_interval==0:
                model.eval()
                avg_loss,accuracy=eval(model,dev_iter)
                " 模型变好，则保存该model的权重参数 "
                if dev_best_loss > avg_loss:
                    dev_best_loss=avg_loss
                    torch.save(model.state_dict(),args.save_path)
                    last_improve = steps
                " 记录最好的模型 ，最后一个模型"
                root_logger.info("\n steps: {}   ,dev_loss {} , dev_acc {} ".format(iter,avg_loss, accuracy))
                logger.add_scalar("dev_loss", avg_loss, global_step=iter)
                logger.add_scalar("dev_accuracy", accuracy, global_step=iter)
def eval(model,eval_iter):
    model.eval()
    loss_all,corrects=0,0
    size=len(eval_iter)
    y_pred_list=np.array([], dtype=int)
    y_true_list=np.array([], dtype=int)
    for steps,(x1,x2,y) in enumerate(eval_iter):
        text,inds,label=x1.to(device),x2.to(device),y.to(device)
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        logit=model(text,inds)
        loss=F.cross_entropy(logit,label)
        loss_all+=loss
        # corrects=(torch.max(logit, 1)[1].view(label.size()).data == label.data).sum()
        y_pred=torch.max(logit,1)[1].cpu().numpy()
        y_pred_list=np.append(y_pred_list,y_pred)  # [0,2,3,...,9] 数量为一个epoch的所有样本数
        y_true_list=np.append(y_true_list,label.cpu().numpy())
    size = len(eval_iter.dataset)
    avg_loss = loss_all/size  # 一个eooch的平均loss值 ，而train_loss是一个batch以内的loss
    accuracy=accuracy_score(y_true_list,y_pred_list)
    t = classification_report(y_true_list, y_pred_list, target_names=['北京', '上海', '成都'])
    print(t)
    # accuracy = 100.0 * float(corrects)/size
    return avg_loss,accuracy
def test(args,model,test_iter):
    model.load_state_dict(torch.load(args.save_path))
    " 把model的数据放到gpu上 "
    model = model.to(device)
    model.eval()
    # start_time = time.time()
    test_loss, test_acc= eval(model, test_iter)
    logging.info("\n test_loss {} , test_acc {} ".format(test_loss, test_acc))
    return test_loss,test_acc