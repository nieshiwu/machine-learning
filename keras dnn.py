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