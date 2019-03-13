#way_1
import numpy
import pandas
import sklearn as sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# calculate time cost
start =time.clock()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# data pre_processing
data = pandas.read_csv(r'F:\TBM_data\2015\data_2015_25\2015_25_shuffle2.csv')
print(data)
dataset = data.values
X = dataset[:,0:data.shape[1]-2].astype(float)
Y = dataset[:,data.shape[1]-1].astype(int)
print(X)
print(Y)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print(encoded_Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=26, activation='relu',kernel_initializer='he_uniform'))
	model.add(Dense(120, activation='relu',kernel_initializer='he_uniform'))
	model.add(Dense(180, activation='relu',kernel_initializer='he_uniform'))
	model.add(Dense(120, activation='relu',kernel_initializer='he_uniform'))
	model.add(Dense(60, activation='relu',kernel_initializer='he_uniform'))
	model.add(Dense(3, activation='softmax',kernel_initializer='he_uniform'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Nadam',metrics=['accuracy'])
	return model

# create estimator
estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=4000)

# spilt data
X_train,X_test,dummy_y_train,dummy_y_test = train_test_split(X, dummy_y, test_size=0.2, shuffle=False)
Y_train,Y_test=train_test_split(Y, test_size=0.2, shuffle=False)

# train the estimator
estimator.fit(X, dummy_y)
#
# test the estimator
loss, accuracy=estimator.model.evaluate(X_test, dummy_y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)

# perdiction the estimator
predictions = estimator.predict(X_test)
answers=encoder.inverse_transform(predictions)
answers=answers.astype(int)
accuracy=sklearn.metrics.classification_report(Y_test,answers)
print(accuracy)


end = time.clock()
print('Running time: %s Seconds'%(end-start))


#way_1
class DNN:
    #initial the network top
    def __init__(self,input_dim,hided1,hided2,hided3,hided4,hided5,output_dim):
        self.input_dim=input_dim
        self.hided1 = hided1
        self.hided2 = hided2
        self.hided3 = hided3
        self.hided4 = hided4
        self.hided5 = hided5
        self.ouput_dim = output_dim

    # define baseline model
    def baseline_model(self):
        # create model
        model = Sequential()
        model.add(Dense(self.hided1, input_dim=self.input_dim, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.hided2, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.hided3, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.hided4, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.hided5, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.ouput_dim, activation='softmax',kernel_initializer='he_uniform'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='Nadam',metrics=['accuracy'])
        return model

class Processing:
    def __init__(self,test_size):
        self.test_size=test_size

    def import_data(self,data):
        X = data.iloc[:,0:data.shape[1]-1]
        Y= data.loc[:,'rank']
        return X,Y

    def transform(self,X,Y,encoder):
        MN = MinMaxScaler()
        X = MN.fit_transform(X)

        # encode class values as integers
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)
        return X,dummy_y

    def split_data(self,X,dummy_y,Y):
        X_train, X_test, dummy_y_train, dummy_y_test ,Y_train, Y_test= train_test_split(X, dummy_y, Y, test_size=self.test_size, random_state=1)
        return X_train, X_test, dummy_y_train, dummy_y_test ,Y_train, Y_test

****************************************************************************************************************************************

#way_2
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler
from numpy import argmax
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data=pd.read_csv(r'C:\Users\user\Desktop\feature_ascending.csv')
data=data.drop(['Unnamed: 0'],axis=1)

#data processing
wash=Processing(0.2)
x,Y=wash.import_data(data)
encoder = LabelEncoder()

totall_loss=list()
totall_accuracy=list()
for i in range(1,x.shape[1]+1):
    X=x.iloc[:,0:i]
    print(X.shape[1],end=' ')

    encoder = LabelEncoder()
    X,dummy_y=wash.transform(X,Y,encoder)



    X_train, X_test, dummy_y_train, dummy_y_test ,Y_train, Y_test=wash.split_data(X,dummy_y,Y)

    # create estimator
    input_dim=X_train.shape[1]
    model=DNN(input_dim,60,120,180,120,60,3)
    model=model.baseline_model()

    #train the DNN
    model.fit(x=X_train,y=dummy_y_train,epochs=30, batch_size=4000,verbose=0)

    # test the estimator
    loss, accuracy=model.evaluate(X_test, dummy_y_test)
    print('test loss: ', loss,end=' ')
    print('test accuracy: ', accuracy)
    totall_loss.append(loss)
    totall_accuracy.append(accuracy)
    # perdiction the estimator
    predictions = model.predict(X_test)
    predictions=argmax(predictions,axis=1)
    answers=encoder.inverse_transform(predictions)
    answers=answers.astype(int)
    accuracy=classification_report(Y_test,answers)
    print(accuracy)
    continue
print(totall_accuracy)
print(totall_loss)
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
plt.plot(totall_accuracy)
ax1.set_ylabel('accuracy')
ax1.set_xlabel('input_dim')
ax2=fig.add_subplot(1,2,2)
plt.plot(totall_loss)
ax2.set_xlabel('input_dim')
ax2.set_ylabel('loss')
plt.show()

#plot the curve between number of input dim with accuracy
plt.plot(totall_loss,label='loss')
plt.plot(totall_accuracy,label='accuracy')
plt.xlabel('number of input_dim')
plt.ylabel('value')
plt.title('DNN')
plt.legend(['loss','accuracy'],loc='best')
plt.show()
