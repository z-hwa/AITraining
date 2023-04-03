#construct develope enviroument
#using python with tensorflow
#since kera is the highest meta framework, we begin from it

#install Anaconda
#install tensorflow rec. GPU version
#pip install keras

#import function
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist #訓練資料的資料庫
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils #用於在後續將label標籤轉為 one-hot-encoding
from matplotlib import pyplot as plt

#載入MNIST資料庫的訓練資料 並自動分為訓練組以及測試組
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#建立簡單的線性執行模型
model = Sequential()
# add input layer, hidden layer which has 256 ouput variable
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
#add output layer
model.add(Dense(units=10, kernel_initializer='normal',activation='softmax'))

#編譯： 選擇損失函數 優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#將training 的label 進行one-hot encoding
#例如數字7經過one-hot encoding 轉換後是 0000001000 即第七個值是1
y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot =  np_utils.to_categorical(y_test)

#將training的input資料轉為2維
x_train_2D = x_train.reshape(60000, 28*28).astype('float32')
x_test_2D = x_test.reshape(10000, 28*28).astype('float32')

x_train_norm = x_train_2D/255
x_test_norm = x_test_2D/255

#進行訓練 訓練過程會存在train_history變數中
train_history = model.fit(x=x_train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)

#顯示訓練成果(分數)
scores = model.evaluate(x_test_norm, y_TestOneHot)
print()
print('\t[Info] Accuracy of testing data = {:2.1f}%'.format(scores[1]*100.0))

#預測
X = x_test_norm[0:10,:]
predictions = np.argmax(model.predict(X), axis=-1)
print(predictions)