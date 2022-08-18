import xlrd,pickle
import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录
sys.path.append(BASE_DIR) #添加环境变量

" 处理 forward_incitions 为空的情况 "
def pro_null(datasets):
    for i in range(len(datasets)):
        if datasets[i]['forward_incitions']=='':
            datasets[i]['forward_incitions']=0
        else:
            datasets[i]['forward_incitions']=int(datasets[i]['forward_incitions'])
    # 处理 back_incitions 为空的情况
    for i in range(len(datasets)):
        if datasets[i]['back_incitions']=='':
            datasets[i]['back_incitions']=0
        else:
            datasets[i]['back_incitions']=int(datasets[i]['back_incitions'])
    # 处理 dep_claims 为空的情况
    for i in range(len(datasets)):
        if datasets[i]['dep_claims']=='':
            datasets[i]['dep_claims']=0
    # 处理 len_claims 为空的情况
    for i in range(len(datasets)):
        if datasets[i]['len_claims']=='':
            datasets[i]['len_claims']=0
    # 处理 num_inventors 为空的情况
    for i in range(len(datasets)):
        if datasets[i]['num_inventors']=='':
            datasets[i]['num_inventors']=0
    # 处理 num_family 为空的情况
    for i in range(len(datasets)):
        if datasets[i]['num_family']=='':
            datasets[i]['num_family']=0
    # 处理 cpcs 为空的情况
    for i in range(len(datasets)):
        if datasets[i]['cpcs']=='':
            datasets[i]['cpcs']=0
    # 处理 ipcs 为空的情况
    for i in range(len(datasets)):
        if datasets[i]['ipcs']=='':
            datasets[i]['ipcs']=0
    return datasets

" 把几个属性 从str转为int "
def str2int(train_datasets):
    for i in range(len(train_datasets)):
        if type(train_datasets[i]['ind_claims'])=='str':
            train_datasets[i]['ind_claims']=int(train_datasets[i]['ind_claims'])
        if type(train_datasets[i]['dep_claims'])=='str':
            train_datasets[i]['dep_claims']=int(train_datasets[i]['dep_claims'])
        if type(train_datasets[i]['len_claims'])=='str':
            train_datasets[i]['len_claims']=int(train_datasets[i]['len_claims'])
    return train_datasets

# 根据 forward_incitions 次数确定k1,k2
def grade_nums(train_datasets,dev_datasets,test_datasets):
    ' 确定总个数m'
    m=len(train_datasets)+len(test_datasets)+len(dev_datasets)
    ' 确定前20%的个数n'
    n1=int(m*0.2)
    ' 确定前20-60%的个数n'
    n2=int(m*0.6)-int(m*0.2)
    ' 确定前60-100%的个数n'
    n2=m-n1-n2
    ' 确定第n大的数'
    arr=[int(train_datasets[i]['forward_incitions']) for i in range(len(train_datasets))]
    arr2=[int(train_datasets[i]['forward_incitions']) for i in range(len(train_datasets))]
    arr3=[int(train_datasets[i]['forward_incitions']) for i in range(len(train_datasets))]
    arr=sorted(arr,reverse=True)
    k1=arr[n1-1]
    k2=arr[n2-1]  
    return k1,k2

" 补充grade 属性 "
def get_grade(train_datasets,k1,k2):
    for i in range(len(train_datasets)):
        if train_datasets[i]['forward_incitions']>=k1:
            train_datasets[i]['grade']=1
        elif train_datasets[i]['forward_incitions']>=k2 and train_datasets[i]['forward_incitions']<k1 :
            train_datasets[i]['grade']=2
        else:
            train_datasets[i]['grade']=3
    return train_datasets

" 得到数据集 "
def init_datasets(path,rows=None):
    #打开excel
    wb = xlrd.open_workbook(path)
    #按工作簿定位工作表
    sh = wb.sheet_by_name('下载的著录项1')
    print(sh.nrows)#有效数据行数
    print(sh.ncols)#有效数据列数
    datas={}
    datasets=[]
    if rows==None:
        until_rows=sh.nrows
    else:
        until_rows=rows
    #遍历excel，打印所有数据
    for i in range(1,until_rows):
        datas={}
        # 如果独立要求数量为0 则跳过这条数据
        if sh.row_values(i)[10]=='':
            continue
        if type(sh.row_values(i)[10])=='str':
            sh.row_values(i)[10]=int(sh.row_values(i)[10])
        if type(sh.row_values(i)[11])=='str':
            sh.row_values(i)[11]=int(sh.row_values(i)[11])
        if type(sh.row_values(i)[13])=='str':
            sh.row_values(i)[13]=int(sh.row_values(i)[13])
        datas['title']=sh.row_values(i)[2]
        datas['abstract']=sh.row_values(i)[3]
        datas['ind_claims']=sh.row_values(i)[10]
        datas['dep_claims']=sh.row_values(i)[11]
        datas['len_claims']=sh.row_values(i)[13]
        datas['len_abstract']=len(sh.row_values(i)[3])
        datas['back_incitions']=sh.row_values(i)[33]
        datas['forward_incitions']=sh.row_values(i)[34]
        datas['num_inventors']=int(sh.row_values(i)[28])
        datas['num_family']=int(sh.row_values(i)[40])
        datas['cpcs']=sh.row_values(i)[20].count(';')+1
        datas['ipcs']=sh.row_values(i)[21].count(';')+1
        datasets.append(datas)
    datasets=pro_null(datasets)
    return datasets
# 保存到pickle文件中，方便下次读取
if __name__ == "__main__":
    train_path='data/1.xls'
    dev_path='data/2.xls'
    test_path='data/7.xls'
    train_datasets=init_datasets(train_path,None)
    dev_datasets=init_datasets(dev_path,4000)
    test_datasets=init_datasets(test_path,3000)
    k1,k2=grade_nums(train_datasets,dev_datasets,test_datasets)

    train_datasets=pro_null(train_datasets)
    train_datasets=str2int(train_datasets)
    train_datasets=get_grade(train_datasets,k1,k2)

    dev_datasets=pro_null(dev_datasets)
    dev_datasets=str2int(dev_datasets)
    dev_datasets=get_grade(dev_datasets,k1,k2)

    test_datasets=pro_null(test_datasets)
    test_datasets=str2int(test_datasets)
    test_datasets=get_grade(test_datasets,k1,k2)

    with open("data/indictors11_dev.pkl","wb") as f:
        pickle.dump(dev_datasets, f)
    with open("data/indictors11_test.pkl","wb") as f:
        pickle.dump(test_datasets, f)
    with open("data/indictors11_train.pkl","wb") as f:
        pickle.dump(train_datasets, f)