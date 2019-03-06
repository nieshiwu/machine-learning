import os
import pandas as pd
import numpy as np
import sklearn

#2015年将原始数据单个特征合并为一个csv文件
def concat_to_one(folder_path,name,save_path):
    filenames = ['5.26-6.1', '6.1-6.7', '6.7-6.14', '6.14-6.21', '6.21-6.28', '6.28-7.5', '7.5-7.8']
    data_all = pd.DataFrame(columns=[name, 'dt'])
    for filename in filenames:
        path=os.path.join(folder_path,filename)+'\\'+name+'.xls'
        data = pd.read_excel(path, usecols=[1, 2], skiprows=[0, 1])
        data.columns = [name, 'dt']
        data_all = data_all.append(data)
    data_all.index = data_all.dt
    data_all.drop(['dt'], axis=1, inplace=True)
    data_all.to_csv(save_path)

names=['TotalAdvForce']
for name in names:
    concat_to_one('E:\\科研\\大数据数据\\吉林引松\\2015_data',name,'C:\\Users\\user\\Desktop'+'\\'+name+'.csv')

#将多个的单个特征csv文件合并为一个csv文件
def merge(AdvSpd_path,folder_path,save_path,first_name):
    data_all=pd.read_csv(AdvSpd_path,engine='python')
    data_all= data_all.drop_duplicates()
    filenames = os.listdir(folder_path)
    for filename in filenames:
        print(filename)
        if filename!=first_name:
            file_path=os.path.join(folder_path,filename)
            data_file=pd.read_csv(file_path,engine='python')
            data_file=data_file.drop_duplicates()
            data_all=pd.merge(data_all,data_file,on='dt',how='inner')
    data_all.to_csv(save_path)

# AdvSpd_path=r'F:\TBM_data\2015\data_2015_more\more\concat_to_one\AdvSpd_more.csv'
# folder_path='F:\\TBM_data\\2015\\data_2015_more\\more\\concat_to_one'
# save_path=r'C:\Users\user\Desktop\2015_more_merge.csv'
# first_name='AdvSpd_more.csv'
# merge(AdvSpd_path,folder_path,save_path,first_name)

#对数据进行清洗
def washing(data,save_path):
    #去除不转的和不掘进的数据
    data = data[data['AdvSpd'] > 0.1]
    data = data[data['CwhRotSpd'] > 0.1]
    # #去除奇异值
    data.iloc[:, 1:data.shape[1]] = data.iloc[:, 1:data.shape[1]][data.iloc[:,1:data.shape[1]] < (
                data.iloc[:, 1:data.shape[1]].mean() + 3 * data.iloc[:, data.shape[1]].std())]
    data.iloc[:, 1:data.shape[1]] = data.iloc[:, 1:data.shape[1]][data.iloc[:,1:data.shape[1]] > (
                data.iloc[:, 1:data.shape[1]].mean() - 3 * data.iloc[:, 1:data.shape[1]].std())]
    # 重新划分index
    data.index = range(0, len(data))
    # 填充空值
    data = data.fillna(method='pad')
    data = data.fillna(method='bfill')
    print(data[data.isnull().values == True])
    # #控制小数点后1位
    data = data.round(decimals=2)
    # 计算掘进里程
    ring = np.cumsum(data.AdvSpd * 3 / 60 / 1000)
    data['ring'] = ring
    data.to_csv(save_path,index=False)

# data=pd.read_csv(r'C:\Users\user\Desktop\2015_more_merge.csv',skiprows=range(1,1945))
# data=data.drop('Unnamed: 0',axis=1)
# save_path=r'C:\Users\user\Desktop\2015_more_washing.csv'
# washing(data,save_path)

#将地质围岩等级与掘进数据对应
def group(data,save_path):
    print(data)
    group = []
    for i in data.ring:
        if i < 243.1:
            group.append(3)
        elif i < 281.1:
            group.append(2)
        elif i < 291.1:
            group.append(3)
        elif i < 299.1:
            group.append(4)
        elif i < 309.1:
            group.append(3)
        elif i < 431.1:
            group.append(2)
        elif i < 444.1:
            group.append(3)
        elif i < 451.3:
            group.append(4)
        elif i < 664.1:
            group.append(2)
        elif i < 681.2:
            group.append(3)
        elif i < 720.8:
            group.append(4)
        elif i < 761.1:
            group.append(3)
        elif i < 826.8:
            group.append(4)
        elif i < 853.1:
            group.append(2)
        elif i < 865.7:
            group.append(4)
        elif i < 876.1:
            group.append(2)
        elif i < 915.7:
            group.append(4)
        elif i < 928.1:
            group.append(2)
        elif i < 1046.1:
            group.append(3)
        elif i < 1065.9:
            group.append(4)
        elif i < 1095.7:
            group.append(3)
        elif i < 1110.1:
            group.append(4)
        else:
            group.append(2)
    data['rank'] = group
    print(data)
    data.to_csv(save_path)

# data = pd.read_csv(r'C:\Users\user\Desktop\2015_more_washing.csv',engine='python')
# save_path=r'C:\Users\user\Desktop\2015_more_group.csv'
# group(data,save_path)

#对数据进行均值-方差处理
def normalization_1(data,save_path):
    data.iloc[:,1:data.shape[1]-2]=(data.iloc[:,1:data.shape[1]] - data.iloc[:,1:data.shape[1]].mean()) / (data.iloc[:,1:data.shape[1]].std())
    # 控制小数点后1位
    data.iloc[:,1:data.shape[1]-2]=data.iloc[:,1:data.shape[1]].round(decimals=2)
    data.to_csv(save_path)

##注简单方法
    #max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    #data=data.apply(max_min_scaler)

#对数据进行归一化处理
def normalization_2(data,save_path):
    data.iloc[:,1:data.shape[1]-2] =(data.iloc[:,1:data.shape[1]]-data.iloc[:,1:data.shape[1]].min())/(data.iloc[:,1:data.shape[1]].max()-data.iloc[:,1:data.shape[1]].min())
    data.iloc[:,1:data.shape[1]-2] =data.iloc[:,1:data.shape[1]].round(decimals=2)
    data.to_csv(save_path)

# data=pd.read_csv(r'C:\Users\user\Desktop\2015_more_group.csv')
# data=data.drop('Unnamed: 0',axis=1)
# save_path=r'C:\Users\user\Desktop\2015_more_normalization2.csv'
# normalization_2(data,save_path)

#打乱处理
def shuffle(data,save_path):
    data=sklearn.utils.shuffle(data)
    data.to_csv(save_path,index=False)

# data=pd.read_csv(r'C:\Users\user\Desktop\2015_more_nor2.csv')
# save_path=r'C:\Users\user\Desktop\2015_more_shuffle.csv'
# shuffle(data,save_path)