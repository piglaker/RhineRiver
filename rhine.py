import random
import pandas as pd
import matplotlib as mpl
import numpy as np
import datetime
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch
import xlrd
import math
from read import readFile


epoch = 200
batch_size = 100
root = "./"
    

def cal(x, y):
    mse = 0
    x, y = np.array(x), np.array(y)
    for i in range(len(x)):
        for j in range(len(x[i])):
            mse += np.square(x[i][j] - y[i][j])
    return mse / len(x)



class lstm(nn.Module):
    def __init__(self, input_size=15, hidden_size=20, num_layer=2):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x1, _ = self.lstm(x)
        x1 = x1[:, -1, :]
        # print(x1.shape)
        out = self.out(x1)

        return out


def train(model, loss, optimizer, x_val, y_val):

    optimizer.zero_grad()

    output = 0

    x = torch.autograd.Variable(x_val, requires_grad=False)

    y = torch.autograd.Variable(y_val, requires_grad=False)

    x = x.cuda()

    fx = model.cuda().forward(x)
        
    output += loss.forward(fx, y.cuda())

    output.backward()

    optimizer.step()

    return fx, output.data


def validation(net, X_test):
    
    ans = []

    for i in X_test:

        p = torch.Tensor(i).unsqueeze(0)
        
        y = predict(net, p)
         
        ans.append(y)
        

    return ans


def predict(model, in_):

    x = torch.autograd.Variable(in_, requires_grad=False)

    x = x.cuda()

    output = model.forward(x)

    output = np.array(output.cpu().detach())

    return output


def make_dataset(batch_size = 15):
    table = readFile()
    dataset = []
    X, Y = [], []
    print(len(table))
    for name_list in table:
        data_x, data_y  = [], []
        for name in name_list:
            y, x = slim_launch(name)
            x, y = np.array(x), np.array(y)
            data_x.append(x)
            data_y.append(y)

        data_x, data_y = np.array(data_x), np.array(data_y)
        
        for i in range(len(name_list) - batch_size):
            t2 = np.array(data_y[i + batch_size])
            t1 = np.array(data_x[i : i + batch_size])

            X.append(t1)
            Y.append(t2)

    return X, Y


def get_division(data, start, t, batch_size):
    """

    :param data:
    :param start:
    :param t:
    :param batch_size:
    :return:(20, 30)
    """
    se = []
    for i in range(t):
        v = np.array(data[start + i : start + i + batch_size] )
        se.append(v) 
    se = np.array(se)
    return se


def slim_launch(name):
    """
        :param data:
        :param x:
        :param track:
        :return:[a,a,...],[a]
    """
    def index_track(data, x, track=False):
        if track:
            for i in range(len(data)):
                if i == x:
                    return np.array([data.iloc[x]['1']])
        else:
            p = []
            for i in range(len(data)):
                if i != x:
                    p.append(np.array(data.iloc[i]['1']))
            return np.array(p)

    data = pd.read_csv('./stations' + name)

    return  index_track(data, 4 ,True), index_track(data, 4, False)


def batch(X, Y, index, batch_size = 10):
    
    #r = random.randint(0, len(X) - batch_size)

    #rand = [random.randint(0, len(X) - 1) for _ in range(0, batch_size)]
        
    batch_x , batch_y = [], []
    
    if index + batch_size > len(X):
        return X[index:], Y[index:]
    else:

        for i in range(index, index + batch_size):
        
            batch_x.append(X[index + i]) 
        
            batch_y.append(Y[index + i])

    return np.array(batch_x), np.array(batch_y)


def app():

    X_train, Y_train  = make_dataset()

    print(len(X_train))
    print(len(Y_train))
    
    #for i in range(len(X_train)):
    #    print(X_train[i].shape)
    #    print(i)
    #    if X_train[i].shape[2] == 14:
    #        peint('111111111111')
    #for i in X_train[8165:]:
    #    print(i.shape)

    model = lstm(input_size=15, hidden_size=30, num_layer=3)

    loss = torch.nn.MSELoss(reduce=True, size_average=True)

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                             amsgrad=False)
    
    X_train, X_test = X_train, X_train[11000:] 

    Y_train, Y_test = torch.Tensor(Y_train), Y_train[11000:]
    
    X_test, Y_test = torch.Tensor(X_test), torch.Tensor(Y_test)

    index = 0

    for i in range(epoch):

        batch_x, batch_y = batch(X_train, Y_train, index, batch_size)

        index += batch_size

        batch_x , batch_y = torch.Tensor(batch_x), torch.Tensor(batch_y)

        print(batch_x.shape, batch_y.shape)
        
        out, loss_ = train(model, loss, optimizer, batch_x, batch_y)
        
        preY = validation(model, X_test)
        
        print(len(preY))

        print("Epoch %d,train_loss = %f,test_mse = %.2f" % (i + 1, loss_ / batch_size, cal(preY, Y_test)))

        if i % 200 == 0:

            torch.save(model, "./models_rhine_" + str(i) + ".pth")


if __name__ == '__main__':
    app()

