# -*- coding:GB2312 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

from networks_lstm import LSTM

config = {
    "font.family":'SimSun',
    "font.size": 10.5,
    "axes.unicode_minus":False
}
rcParams.update(config)

def sliding_windows(data, seq_length=4):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

def train(num_epochs):
    train_size = int(len(y) * 0.67)
    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    criterion = torch.nn.MSELoss()    
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, trainY)
        
        loss.backward()
        
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

def predict(model):
    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))
    train_size = int(len(y) * 0.67)

    model.eval()
    train_predict = model(dataX)

    data_predict = train_predict.data.numpy()
    dataY_plot = dataY.data.numpy()

    data_predict = sc.inverse_transform(data_predict)
    dataY_plot = sc.inverse_transform(dataY_plot)

    # plt.plot(dataY_plot)
    # plt.plot(data_predict)
    # plt.xlabel("时间步长")
    # plt.ylabel("电价/元")
    # plt.show()
    return data_predict # numpy list


lstm = LSTM()
sc = MinMaxScaler()
training_raw = pd.read_csv('data/2019-04.csv')
training_data = sc.fit_transform(training_raw)
x, y = sliding_windows(training_data)

# n_epoch = 2000
# train(n_epoch)
# torch.save(lstm, "model/lstm/lstm_{}.pt".format(n_epoch))

model = torch.load("model/lstm/lstm_2000.pt")
predict_data = predict(model)
