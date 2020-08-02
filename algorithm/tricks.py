#https://blog.csdn.net/MiaoB226/article/details/88210189博客学习

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import copy

#1. 保存网络训练最好的权重
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
# 保存测试阶段，准确率最高的模型
if phase == 'val' and epoch_acc > best_acc:
best_acc = epoch_acc
best_model_wts = copy.deepcopy(model.state_dict())
model.load_state_dict(best_model_wts)# 最后网络导入最好的网络权重

# 保存网络模型结构并加载，model.state_dict()里面是权重值
torch.save(model.state_dict(), 'model_state//' + str(epoch) + '_model.pkl')
model.load_state_dict(torch.load('model_AlexNet.pkl')

#2. 锁住网络某几层的权重，只更新某几层权重
from torch.optim import lr_scheduler
if __name__ ==  '__main__':
 
    # 导入Pytorch中自带的resnet18网络模型
    model_ft = models.resnet18(pretrained=True)
    # 将网络模型的各层的梯度更新置为False
    for param in model_ft.parameters():
        param.requires_grad = False
 
    # 修改网络模型的最后一个全连接层
    num_ftrs = model_ft.fc.in_features# 获取最后一个全连接层的输入通道数
    model_ft.fc = nn.Linear(num_ftrs, 2) # 修改最后一个全连接层的的输出数为2
    
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
    
    # 定义学习率的更新方式，每5个epoch修改一次学习率
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    
    #以上进行的是网络上的参数设置，接下来在for i in epoch中训练网络模型
    model_ft = train(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

#3. 修改网络层结构，自己加新的层
import torch.nn as nn
model= models.resnet18(pretrained=True)
model=list(model.children())
model=model[:-1]#最后一层网络不要
print(model[-1].out_features)#可查看最后一层的输出
model.append(nn.Linear(3, 2))#自己新加一层
print(model)

#4. training计时
import time
since = time.time()
    pass
time_elapsed = time.time() - since
print('Training complete in {:.0f}min{:.0f}sec'.format(time_elapsed // 60, time_elapsed % 60))

#5. with torch.no_grad():
#训练阶段需要保存网络各层的梯度，根据梯度和Loss值来更新网络的权重。
#由于验证与测试阶段是不需要更新权重的，所以不用保存梯度，但是Pytorch如果没有在代码中指定不保存梯度的话，默认是保存验证和测试阶段的梯度，这个梯度是十分占显存的。因此，我们只需将测试和验证阶段设置为不保存梯度即可。
       
#6. 
