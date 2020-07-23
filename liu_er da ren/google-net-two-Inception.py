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

class InceptionA(torch.nn.Module):#inception网络要保证b,w,h一样，c可以不同，下例就是c=24*3+16=88
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.branch1x1=nn.Conv2d(in_channels,16,kernel_size=1)

        self.branch5x5_1=nn.Conv2d(in_channels,16,kernel_size=1)#最后卷积核为5x5的那一分支的两个过程
        self.branch5x5_2=nn.Conv2d(16,24,kernel_size=5,padding=2)

        self.branch3x3_1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2=nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3=nn.Conv2d(24,24,kernel_size=3,padding=1)

    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)







class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(88,20,kernel_size=5)

        self.incep1=InceptionA(in_channels=10)
        self.incep2=InceptionA(in_channels=20)

        self.mp=nn.MaxPool2d(2)
        self.fc=nn.Linear(1408,10)


    def forward(self,x):
        in_size=x.size(0)
        x=F.relu(self.mp(self.conv1(x)))#给self.conv1实例化的对象，赋值x
        x=self.incep1(x)
        x=F.relu(self.mp(self.conv2(x)))
        x=self.incep2(x)
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