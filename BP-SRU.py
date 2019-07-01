import pandas as pd

data = pd.read_table('SRU_data.txt', sep='\s+')
# data = np.loadtxt('debutanizer_data.txt', delimiter='\t')
print('data.shape = ', data.shape, '\n')
# print(data.head())

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

device = torch.device('cpu')

data = np.array(data)
x = data[:, 0:len(data[0])-2]
y = data[:, len(data[0])-1]

# print(x[0, :])
# print(x.shape)
# print(y.shape)

untrainInputdata = np.zeros(shape=[10071, 20], dtype=float)
targetOutputdata = np.zeros(shape=[10071, 1])

for i in range(9, 10080):
    untrainInputdata[i-9, :] = [x[i, 0], x[i-5, 0], x[i-7, 0], x[i-9, 0],
                                x[i, 1], x[i-5, 1], x[i-7, 1], x[i-9, 1],
                                x[i, 2], x[i-5, 2], x[i-7, 2], x[i-9, 2],
                                x[i, 3], x[i-5, 3], x[i-7, 3], x[i-9, 3],
                                x[i, 4], x[i-5, 4], x[i-7, 4], x[i-9, 4]]
    targetOutputdata[i-9] = y[i]

x1 = untrainInputdata[0:8000, :]
y1 = targetOutputdata[0:8000]
x2 = untrainInputdata[8000:len(untrainInputdata), :]
y2 = targetOutputdata[8000:len(targetOutputdata)]

plt.plot(y1)
plt.show()

# print(y2.shape)
# print(y1[0:9])

x_train = torch.from_numpy(x1)
x_train = x_train.float()
y_train = torch.from_numpy(y1)
y_train = y_train.view(8000,1)
y_train = y_train.float()
x_test = torch.from_numpy(x2)
x_test = x_test.float()
y_test = torch.from_numpy(y2)
y_test = y_test.view(2071,1)
y_test = y_test.float()
print(x_train.shape)
print(y_train.shape)
# input, output = Variable(x_train), Variable(y_train)

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)
#
#     def forword(self, input):
#         h_relu = F.relu(self.hidden(input))
#         y_pred = self.predict(h_relu)
#         return y_pred
#
# net = Net(7, 28, 1)
# print(net)

# torch.manual_seed(1024)    # reproducible

torch.manual_seed(1024)

net1 = torch.nn.Sequential(
    torch.nn.Linear(20, 60),
    torch.nn.ReLU(),
    torch.nn.Linear(60, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, 1),
).to(device)
print(net1)

learning_rate = 0.2
optimizer = torch.optim.SGD(net1.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()

print(input)

#plt.ion()
plt.figure()
a = [0]
a_now = 0
b = [0]

# def train():
#     net1.train()
for t in range(5000):
    y_pred = net1(x_train)
    loss = loss_func(y_pred, y_train)
    #print(t, loss.item())

    #plt.clf()  # 清空画布所有内容
    a_now = t
    a.append(a_now)  #模拟数据增量流入，保存历史数据
    b.append(loss.item())
    #plt.plot(a, b, 'r')
    #plt.draw()  #注意此函数需要调用
    #time.sleep(0.01)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())

plt.figure(1)
plt.plot(a, b, 'r')

# def test():
#     net1.eval()
#     pred_mseloss = [0]
#     for t in range(894):
#         y_pred = net1(x_test)
#         loss = loss_func(y_pred, y_test)
#         print(t, loss.item())
#         pred_mseloss.append(loss.item())
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(torch.mean(pred_mseloss).float())

# train()
# test()

net1.eval()

y_test1 = net1(x_test)
test_loss = loss_func(y_test1, y_test)
print(test_loss)

u = y_test1
v = y_test
u = u.detach().numpy()
v = v.detach().numpy()
loss_mean = u - v
loss_mean = np.mean(loss_mean)
print('\n loss_mean', loss_mean)
# print(y_test)
# print(y_test1)

y_test.view(1, 2071)
y_test1.view(1, 2071)

y_test = y_test.numpy()
y_test1 = y_test1.detach().numpy()

plt.figure(2)
plt.subplot(211)
plt.title('ANN')
plt.plot(y_test1, 'r-')
plt.plot(y_test)
# plt.axis([0, 2071, -0.05, 0.95])
plt.legend(['Predicted Output', 'Test Data'])

plt.subplot(212)
plt.plot(y_test1 - y_test, 'r-')
# plt.axis([0, 2071, -0.15, 0.15])
plt.legend(['Error Analysis', 'Test Data'])

plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test1, y_test)
print('\nMSE:', mse)

from sklearn.metrics import r2_score
print('相关系数：', r2_score(y_test, y_test1))