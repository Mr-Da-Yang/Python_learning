#google-net 使用两个inception模块实现mnist多分类任务
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
batch_size = 64

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))#你均值和方差都只传入一个参数，就报错了.
    # 这个函数的功能是把输入图片数据转化为给定均值和方差的高斯分布，使模型更容易收敛。图片数据是r,g,b格式，对应r,g,b三个通道数据都要转换。
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                                train=True,
                                download=True,
                                transform=transform)
train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/',
                                train=False,
                                download=True,
                                transform=transform)
test_loader = DataLoader(test_dataset,
                        shuffle=False,
                        batch_size=batch_size)

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels=channels
        self.conv1=nn.Conv2d(channels,channels,
                             kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(channels,channels,
                             kernel_size=3,padding=1)

    def forward(self, x):
        y=F.relu(self.conv1(x))
        y=self.conv2(y)
        return F.relu(x+y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,16,kernel_size=5)
        self.conv2=nn.Conv2d(16,32,kernel_size=5)
        self.mp=nn.MaxPool2d(2)

        self.rblock1=ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc=nn.Linear(512,10)

    def forward(self,x):
        in_size=x.size(0)
        x=self.mp(F.relu(self.conv1(x)))
        x=self.rblock1(x)
        x=self.mp(F.relu(self.conv2(x)))
        x=self.rblock2(x)
        x=x.view(in_size,-1)
        x=self.fc(x)
        return x

model = Net()
device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx , data in enumerate(train_loader, 0):
        inputs,target=data
        inputs,target = inputs.to(device),target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)#outputs:64*10,行表示对于图片的预测，batch=64
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx %300 ==299:
            print('[%d,%5d] loss: %.3f'%(epoch+1,batch_idx+1,running_loss/2000))
            running_loss=0.0


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
    print('Accuracy on test set:%d %% [%d/%d]'%(100*correct/total,correct,total))


if __name__ =="__main__":
    for epoch in range(10):
        train(epoch)#封装起来，若要修改主干就很方便
        test()


