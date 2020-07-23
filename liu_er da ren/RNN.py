# #dataset被压缩成（seq，b，x），然后经过遍历得到（b,x）,将这一序列的值输出
# import torch
# import torch.nn as nn
# batch_size=1
# seq_len=3
# input_size =4
# hidden_size =2
#
# cell =torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)
#
# dataset = torch.randn(seq_len,batch_size,input_size)
# hidden = torch.zeros(batch_size,hidden_size)
#
# for idx, input in enumerate(dataset):
#     print('='*20,idx,'='*20)
#     print(input)
#     print('input size: ',input.shape)
#
#     hidden=cell(input,hidden)
#     print("output size: ",hidden.shape)
#     print(hidden)
# print('='*20,'='*20)
# print(dataset)
# print(dataset.shape)

import torch
import torch.nn as nn
batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1
cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
num_layers=num_layers, batch_first=True)
# (seqLen, batchSize, inputSize)
inputs = torch.randn(batch_size, seq_len, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)
out, hidden = cell(inputs, hidden)
print('Output size:', out.shape)
print('Output:', out)
print('Hidden size: ', hidden.shape)
print('Hidden: ', hidden)