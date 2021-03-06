import pandas as pd

data = pd.read_table('SRU_data.txt', sep='\s+')
# data = np.loadtxt('debutanizer_data.txt', delimiter='\t')
print('data.shape = ', data.shape, '\n')
# print(data.head())

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

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

x_train = x_train.cuda()
y_train = y_train.cuda()
x_test = x_test.cuda()
y_test = y_test.cuda()

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
# TIME_STEP = 5      # rnn time step
INPUT_SIZE = 20      # rnn input size
OUTPUT_SIZE = 1
LR = 0.001          # learning rate
hidden_dim = 15
layer_dim = 1
batch_size = 100
test_samples = 2071
Drop_out = 0.0

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        # Hidden Dimension
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=hidden_dim,       # rnn hidden unit (The number of features in the hidden state h) 这个参数表示的是用于记忆和储存过去状态的节点个数。
            num_layers=layer_dim,         # number of rnn layer (Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1)
            batch_first=False,    # If Ture : input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=Drop_out,            #非0小数，每个rnn输出后加一个drop out 层， 最后一层除外
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        # r_out, (h_n, c_n) = self.lstm(x)

        # Initializing the hidden state with zeros
        # (input, hx, batch_sizes)
        h0 = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim)).cuda()

        c0 = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim)).cuda()

        # One time step (the last one perhaps?)
        r_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # outs = []    # save all predictions
        # for time_step in range(r_out.size(1)):    # calculate output for each time step
        #     outs.append(self.out(r_out[:, time_step, :]))
        # return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 15)
        outs = self.out(r_out[:,-1,:])
        return outs

rnn = LSTM()
rnn.cuda()
print(rnn)

# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=1e-5)   # optimize all cnn parameters
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

# h_state = None      # for initial hidden state

#h_state = np.zeros()

# plt.figure(1, figsize=(12, 5))
# plt.ion()           # continuously plot

# plt.figure()
# a = [0]
# a_now = 0
# b = [0]

num_epochs = 1000
num_samples = 8000
train_loss = []
train_iter = []
n_iter = 0
x_train1 = x_train[:, np.newaxis]
x_test1 = x_test[:, np.newaxis]

test_loss = []
test_iter = []

# for i in range(8000):
#
#     x_train1 = x_train[:, np.newaxis]  # shape (batch, time_step, input_size)
#     # y_train1 = y_train[:, np.newaxis]
#
#     # x_train1 = torch.from_numpy(x_train[np.newaxis, :, np.newaxis])    # shape (batch, time_step, input_size)
#     # y_train1 = torch.from_numpy(y_train[np.newaxis, :, np.newaxis])
#
#     prediction = rnn(x_train1)   # rnn output
#     # !! next step is important !!
#     # h_state = h_state.data        # repack the hidden state, break the connection from last iteration
#
#     loss = loss_func(prediction, y_train)           # calculate loss
#
#     optimizer.zero_grad()                           # clear gradients for this training step
#     loss.backward()                                 # backpropagation, compute gradients
#     optimizer.step()                                # apply gradients

print("starting to train the model for {} epochs!".format(num_epochs))
for epoch in range(num_epochs):
    for i in range(0, int(num_samples/batch_size -1)):


        features = Variable(x_train1[i*batch_size:(i+1)*batch_size, :])
        Kt_value = Variable(y_train[i*batch_size:(i+1)*batch_size])

        #print("Kt_value={}".format(Kt_value))

        optimizer.zero_grad()

        prediction = rnn(features)
        #print("outputs ={}".format(outputs))

        loss = loss_func(prediction, Kt_value)

        train_loss.append(loss.data)
        train_iter.append(n_iter)

        #print("loss = {}".format(loss))
        loss.backward()

        optimizer.step()

        n_iter += 1
        test_batch_mse =list()
        if n_iter%100 == 0:
            for i in range(0, int(test_samples/batch_size -1)):
                features = Variable(x_test1[i*batch_size:(i+1)*batch_size, :])
                Kt_test = Variable(y_test[i*batch_size:(i+1)*batch_size])

                prediction = rnn(features)

                Kt_test.data = Kt_test.data.cpu()
                prediction.data = prediction.data.cpu()
                test_batch_mse.append(np.mean([(Kt_test.data.numpy() - prediction.data.numpy().squeeze())**2],axis=1))

            test_iter.append(n_iter)
            test_loss.append(np.mean([test_batch_mse], axis=1))

            print('Epoch: {} Iteration: {}. Train_MSE: {}. '.format(epoch, n_iter, loss.data))
            # print('Epoch: {} Iteration: {}. Train_MSE: {}. Test_MSE: {}'.format(epoch, n_iter, loss.data, test_loss[-1]))

figLossTrain = plt.figure()
plt.plot(np.array(test_loss).squeeze(), 'r')




# rnn.eval()
#
x_test1 = x_test[:, np.newaxis]  # shape (batch, time_step, input_size)
# y_test1 = y_test[:, np.newaxis]

y_test1= rnn(x_test1)
test_loss = loss_func(y_test1, y_test)
print(test_loss)

y_test.view(1, 2071)
y_test1.view(1, 2071)

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