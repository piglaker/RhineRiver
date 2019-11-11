import pandas as pd
import matplotlib as mpl
import numpy as np
import datetime
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch
import keras
from keras import models
from keras import layers
import tensorflow as tf
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

SIZE_OF_WINDOW = 30



def cal(x, y):
    mse = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            mse += np.square(x[i][j] - y[i][j])
    return mse / x.shape[0]



def train(model, loss, optimizer, x_val, y_val):
    x = torch.autograd.Variable(x_val, requires_grad=False)
    y = torch.autograd.Variable(y_val, requires_grad=False)
    x = x.cuda()
    optimizer.zero_grad()
    #print(x.shape, y.shape)
    fx = model.cuda().forward(x)
    #print(fx.shape)
    output = loss.forward(fx, y.cuda())
    output.backward()
    optimizer.step()
    return output.data

def predict(model, x_val, batch_size):
    num = x_val.shape[0] // 20
    print(x_val.shape)
    #print(x_val.shape[0])
    x = torch.autograd.Variable(x_val, requires_grad=False)
    x = x.cuda()
    #print(x.shape)
    output = model.forward(x[batch_size * 0: batch_size * (0 + 1)])
    for i in range(1, num):
        #print(x[batch_size * num: batch_size * (num + 1)].shape)
        output = torch.cat((output, model.forward(x[batch_size * i: batch_size * (i + 1)])),0)
    output = np.array(output.cpu().detach())
    #print(output.shape)
    return output#.cpu().data.numpy()


def generate_data(data, start, end, SIZE_OF_WINDOW):
    dataX = []
    dataY = []
    for i in range(end - start + 2 - SIZE_OF_WINDOW):
        dataX.append(np.asarray(data[start + i : start + i + SIZE_OF_WINDOW - 3][["discharge","water_level"]]))
        dataY.append(np.asarray(data[start + i + 27 : start + i + SIZE_OF_WINDOW]["water_level"]))
    return dataX, dataY


data = pd.read_csv("./StudentData/river_data.csv")
#x = np.linspace(0, 1,14893)
#print(len(x))
data1 = data.where(data["station_no"] == 6335020).dropna(axis = 0)
print(len(data1))
data_x = pd.read_csv("./StudentData/to_predict.csv")
date = data_x["date"]
new_date = []
for i in range(len(date)//2):
    if i % 3 == 0:
        d = datetime.datetime.strptime(date.ix[i], '%Y-%m-%d')
        new = (d+datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
        new_date.append(new)
#print(new_date)
date = []
for i in range(len(new_date)):
    date.append(int(data1.loc[data1["date"] == new_date[i]].index.values))
#print(date)
X = []
Y = []
for i in range(len(date) - 1):
    dataX, dataY = generate_data(data1, date[i], date[i + 1], SIZE_OF_WINDOW)
    for j in range(len(dataX)):
        X.append(dataX[j])
        Y.append(dataY[j])

print(len(X))    


X = np.asarray(X)
Y = np.asarray(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)







filepath = "weights-improvement-{epoch:02d}.hdf5"
model = models.Sequential()
model.add(layers.CuDNNLSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(layers.Dense(3))
model.compile(loss='mse', optimizer='adam')
# fit network
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,mode='min')
callbacks_list = [checkpoint]
history = model.fit(X_train, y_train, epochs=5000, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False, callbacks=callbacks_list)
