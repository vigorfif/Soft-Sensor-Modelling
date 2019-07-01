import pandas as pd

data = pd.read_table('SRU_data.txt', sep='\s+')
# data = np.loadtxt('debutanizer_data.txt', delimiter='\t')
print('data.shape = ', data.shape, '\n')
# print(data.head())

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')

data = np.array(data)
x = data[:, 0:len(data[0])-2]
y = data[:, len(data[0])-1]

# print(x[0, :])
# print(x.shape)
# print(y.shape)

# untrainInputdata = np.zeros(shape=[10071, 5], dtype=float)
# targetOutputdata = np.zeros(shape=[10071, 1])
#
# for i in range(1, 10080):
#     untrainInputdata[i-9, :] = [x[i, 0], x[i-5, 0], x[i-7, 0], x[i-9, 0],
#                                 x[i, 1], x[i-5, 1], x[i-7, 1], x[i-9, 1],
#                                 x[i, 2], x[i-5, 2], x[i-7, 2], x[i-9, 2],
#                                 x[i, 3], x[i-5, 3], x[i-7, 3], x[i-9, 3],
#                                 x[i, 4], x[i-5, 4], x[i-7, 4], x[i-9, 4]]
#     targetOutputdata[i-9] = y[i]

# x1 = untrainInputdata[0:8000, :]
# y1 = targetOutputdata[0:8000]
# x2 = untrainInputdata[8000:len(untrainInputdata), :]
# y2 = targetOutputdata[8000:len(targetOutputdata)]

x1 = x[0:8000, :]
y1 = y[0:8000]
x2 = x[8000:len(x), :]
y2 = y[8000:len(y)]

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
y_test = y_test.view(2080,1)
y_test = y_test.float()
print(x_train.shape)
print(y_train.shape)

x_train = x_train.cuda()
y_train = y_train.cuda()
x_test = x_test.cuda()
y_test = y_test.cuda()

torch.manual_seed(1024)    # reproducible

# Hyper Parameters
TIME_STEP = 5      # rnn time step
INPUT_SIZE = 5      # rnn input size
LR = 0.3           # learning rate


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=50,     # rnn hidden unit (The number of features in the hidden state h) 这个参数表示的是用于记忆和储存过去状态的节点个数。
            num_layers=1,       # number of rnn layer (Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1)
            nonlinearity='relu',  #用relu代替tanh
            batch_first=False,   # If Ture : input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(50, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        # outs = []    # save all predictions
        # for time_step in range(r_out.size(1)):    # calculate output for each time step
        #     outs.append(self.out(r_out[:, time_step, :]))
        # return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        r_out = r_out.view(-1, 50)
        outs = self.out(r_out)
        return outs, h_state


rnn = RNN()
rnn.cuda()
print(rnn)

optimizer = torch.optim.SGD(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state

# plt.figure(1, figsize=(12, 5))
# plt.ion()           # continuously plot

for i in range(2000):

    x_train1 = x_train[:, np.newaxis]  # shape (batch, time_step, input_size)
    # y_train1 = y_train[:, np.newaxis]

    # x_train1 = torch.from_numpy(x_train[np.newaxis, :, np.newaxis])    # shape (batch, time_step, input_size)
    # y_train1 = torch.from_numpy(y_train[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x_train1, h_state)   # rnn output
    # !! next step is important !!
    h_state = h_state.data        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y_train)           # calculate loss
    optimizer.zero_grad()                           # clear gradients for this training step
    loss.backward()                                 # backpropagation, compute gradients
    optimizer.step()                                # apply gradients

    # plotting
    # plt.cla
    # plt.plot(y_train.data.numpy().flatten(), 'r-')
    # plt.plot(prediction.data.numpy().flatten(), 'b-')
    # plt.draw(); plt.pause(0.05)

rnn.eval()

x_test1 = x_test[:, np.newaxis]  # shape (batch, time_step, input_size)
# y_test1 = y_test[:, np.newaxis]

print(x_test1.shape)

y_test1, h_state = rnn(x_test1, h_state.data)

# print(y_test1.shape)
# print(y_test1[0:5])

print(y)

test_loss = loss_func(y_test1, y_test)
print(test_loss)

y_test.view(1, 2080)
y_test1.view(1, 2080)

y_test = y_test.cpu()
y_test1 = y_test1.cpu()

y_test = y_test.numpy()
y_test1 = y_test1.detach().numpy()

plt.subplot(211)
plt.title('RNN')
plt.plot(y_test1, 'r-')
plt.plot(y_test)
# plt.axis([0, 895, -0.05, 0.95])
plt.legend(['Predicted Output', 'Test Data'])

plt.subplot(212)
plt.plot(y_test1 - y_test, 'r-')
# plt.axis([0, 895, -0.15, 0.15])
plt.legend(['Error Analysis', 'Test Data'])

plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test1, y_test)
print('\nMSE:', mse)

from sklearn.metrics import r2_score
print('相关系数：', r2_score(y_test, y_test1))