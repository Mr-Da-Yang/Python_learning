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


#2. 微调网络
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
#3. 计时
import time
since = time.time()
    pass
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
