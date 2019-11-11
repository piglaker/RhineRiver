import pandas as pd
import matplotlib as mpl
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle 
import os
import math
import pandas as pd
import matplotlib as mpl
import numpy as np
import datetime
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch


def cal(x, y):
    mse = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            mse += np.square(x[i][j] - y[i][j])
    return mse / x.shape[0]


def train_model(model, loss, optimizer, x_val, y_val):
    x = torch.autograd.Variable(x_val, requires_grad=False)
    y = torch.autograd.Variable(y_val, requires_grad=False)
    #print(x.shape)
    x = x.cuda()
    optimizer.zero_grad()

    fx = model.cuda().forward(x)
    output = loss.forward(fx, y.cuda())

    output.backward()
    optimizer.step()
    return output.data


def predict(model, x_val, batch_size):
    num = x_val.shape[0] // 20
    #print(x_val.shape)
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


SIZE_OF_WINDOW = 50
root = "./StudentData/"
PIC_PATH = '../ljj/pickle'

def generate_data(data, start, end, SIZE_OF_WINDOW):
    begin = data.index.values[0]
    train = []
    test = []
    for i in range(1, end - start + 2 - SIZE_OF_WINDOW):
        X = data[-begin + start + i : -begin + start + i + SIZE_OF_WINDOW - 3][["discharge_delta", "water_delta"]]
        #print(data[-begin + start + i : -begin + start + i + SIZE_OF_WINDOW - 3])
        X['Wea'] = 0
        X = np.array(X)
        for j in range(len(X)):
            k = pickle.load(open(os.path.join(PIC_PATH,str(data.iloc[-begin + start + i +j]['date'])+'.pi'), "rb+"))
            if not math.isnan(k.predict([50.937207, 6.965515])):
                #print(k.predict([51.22629, 6.76808]))
                X[j,2] = k.predict([50.937207, 6.965515])
        
        #print(X)
        train.append(X)
        test.append(np.array(data[-begin + start + i + SIZE_OF_WINDOW - 3: -begin + start + i + SIZE_OF_WINDOW]["water_delta"]))
    return train, test


data = pd.read_csv(root + "river_data.csv")
data_x = pd.read_csv(root + "to_predict.csv")


data1 = data.where(data["station_no"] == 6335060).dropna(axis = 0)


#find the date
date = data_x["date"]
new_date = []
for i in range(len(date)//2):
    if i % 3 == 0:
        d = datetime.datetime.strptime(date.ix[i], '%Y-%m-%d')
        new = (d+datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
        new_date.append(new)
     

    
#the index of date
a = data1.index.values[0]
date = [a]
for i in range(len(new_date)):
    date.append(int(data1.loc[data1["date"] == new_date[i]].index.values))

    
# generate dataset
X = []
Y = []

data1['wa1']=data1['water_level'].shift(1)
data1['water_delta'] = -data1['wa1'] + data1['water_level']
data1['dis1']=data1['discharge'].shift(1)
data1['discharge_delta'] = -data1['dis1'] + data1['discharge']

for i in range(len(date) - 1):
    print(i,'/',str((len(date) - 1)))
    train, test = generate_data(data1, date[i], date[i + 1], SIZE_OF_WINDOW)
    for j in range(len(train)):
        X.append(train[j])
        Y.append(test[j])

print(len(X))

pickle.dump(X, open('x_delta_60_.pkl', 'wb'))
pickle.dump(Y, open('y_delta_60.pkl', 'wb'))
#X = pickle.load(open('x_delta_60.pkl', 'rb+'))
#Y = pickle.load(open('y_delta_60.pkl', 'rb+'))


class lstm(nn.Module):
    def __init__(self,input_size = 3, hidden_size = 200, num_layer=2):
        super(lstm,self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layer) 
        self.out = nn.Linear(hidden_size,3) 
    def forward(self,x):
        x1, _ = self.lstm(x)
        #print(x1.shape)
        x1 = x1[:,-1,:]
        #print(x1.shape)
        out = self.out(x1) 
 
        return out

X = np.asarray(X)
Y = np.asarray(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)



n_examples = len(X_train)
model = lstm(input_size = 3, hidden_size = 200, num_layer=3)
loss = torch.nn.MSELoss(reduce = True, size_average=True)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
batch_size = 20
epochs = 5000
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
#X_train = torch.unsqueeze(X_train, 1)
#X_test = torch.unsqueeze(X_test, 1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
best = 100000000
save = []
for i in range(epochs):
    cost = 0
    num_batches = n_examples / batch_size
    for k in range(int(num_batches)):
        start, end = k * batch_size, (k + 1) * batch_size
        cost += train_model(model, loss, optimizer, X_train[start:end, :], y_train[start:end, :])
    predY = predict(model, X_test, batch_size)
    #print(predY.shape, y_test.shape)
    best1 = cal(predY , y_test)
    print("Epoch %d,loss = %f,mse = %.2f" % (i + 1, cost / num_batches, best1 ))
    save.append([cost / num_batches, cal(predY , y_test)])
    if best1 < best:
        best = best1
        torch.save(model, "./models_" + str(SIZE_OF_WINDOW) +"attention_delta_60.pth")
        d = np.array(save)
        np.savetxt("./save_lstm3_discharge.txt", d)
