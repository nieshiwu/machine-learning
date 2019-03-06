import numpy as np
import pandas as pd
import pydotplus
import sklearn as sklearn
from sklearn.model_selection import train_test_split
# pd.set_option('display.max_columns', None)
data = pd.read_csv(r'F:\TBM_data\2015\data_2015_40\2015_41_group.csv')
data_1=data.drop(['dt','RstPumpPrs','rank','ring'],axis=1)
data_1['rank']=data.loc[:,'rank']
x=data_1.iloc[:,0:data_1.shape[1]-1]
y=data_1.iloc[:,data_1.shape[1]-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
clf=sklearn.tree.DecisionTreeClassifier(splitter="random",max_depth=15,min_samples_split=25,min_samples_leaf=12)
clf=clf.fit(x_train,y_train)
columns=data_1.columns.values
columns=np.delete(columns,columns.shape[0]-1)
print(columns)
importance=pd.DataFrame({'name':columns,'value':clf.feature_importances_})
importance_sort=importance.sort_values(by='value',ascending=False)
print(importance_sort)