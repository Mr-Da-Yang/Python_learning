'''
mnist多分类任务one-hot
1.total_training_time
2.save models.state_dict()
'''

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
batch_size = 64

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))#你均值和方差都只传入一个参数，就报错了.
    # 这个函数的功能是把输入图片数据转化为给定均值和方差的高斯分布，使模型更容易收敛。图片数据是r,g,b格式，对应r,g,b三个通道数据都要转换。
])

train_dataset = datasets.MNIST(root='./dataset',
                                train=True,
                                download=True,
                                transform=transform)
train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=batch_size)
test_dataset = datasets.MNIST(root='./dataset',
                                train=False,
                                download=True,
                                transform=transform)
test_loader = DataLoader(test_dataset,
                        shuffle=False,
                        batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(320,10)

    def forward(self,x):
        batch_size=x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size,-1)
        x = self.fc(x)
        return x

model = Net()
device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    since=time.time()
    running_loss = 0.0
    loss_list, batch_list = [], []

    for batch_idx , data in enumerate(train_loader, 0):
        inputs,target=data
        inputs,target = inputs.to(device),target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)#outputs:64*10,行表示对于图片的预测，batch=64
        loss = criterion(outputs,target)

        #存储每一次的梯度与迭代次数
        loss_list.append(loss.detach().cpu().item())
        batch_list.append(batch_size + 1)

        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx %300 ==299:
            print('[%d,%5d] loss: %.3f'%(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0

    # 保存网络模型结构
    torch.save(model.state_dict(), 'model_state//' + str(epoch) + '_model.pkl')
    time_elapsed = time.time() - since
    print('{}_epo_training complete in {:.0f}min{:.0f}sec'.format(str(epoch),time_elapsed // 60, time_elapsed % 60))#验证过
    return time_elapsed#这里返回一个epoch的运行时间

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data,dim=1)
            total+=labels.size(0)#每一批=64个，所以total迭代一次加64
            correct +=(predicted==labels).sum().item()
    print('Accuracy on test set:%d %%'%(100*correct/total))


if __name__ =="__main__":

    total_training_time=0
    for epoch in range(10):
        print("-"*50,'epoch',epoch+1,"-"*50)
        time_elapsed=train(epoch)#封装起来，若要修改主干就很方便
        total_training_time+=time_elapsed
        feature_extractor = model
        # test()
    print('total_training complete in {:.0f}min{:.0f}sec'.format( total_training_time // 60, total_training_time % 60))

