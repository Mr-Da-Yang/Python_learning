"""
导入自己的模型，训练并保存错误率最小时的网络参数
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

import CAM.model
import CAM.utils

def train(config):
    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)

    train_loader, num_class = CAM.utils.get_trainloader(config.dataset,
                                        config.dataset_path,
                                        config.img_size,
                                        config.batch_size)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CAM.model.CNN(img_size=config.img_size, num_class=num_class).to(device)#先定一个cnn类
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=config.lr)
    min_loss = 999


    print("START TRAINING")
    for epoch in range(config.epoch):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):#len(train_loader)=1563
            #print(images, labels)#(torch.Size([32, 3, 128, 128]), torch.Size([32]))

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = cnn(images)#将images输入到cnn中，outputs是一个10分类torch.Size([32, 10])， _ 是最后一个conv输出的特征torch.Size([32, 10, 16, 16])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % config.log_step == 0:
                if config.save_model_in_epoch:#这里设置为False
                    torch.save(cnn.state_dict(), os.path.join(config.model_path, config.model_name))
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                    % (epoch + 1, config.epoch, i + 1, len(train_loader), loss.item()))

        avg_epoch_loss = epoch_loss / len(train_loader)
        print('Epoch [%d/%d], Loss: %.4f'
                    % (epoch + 1, config.epoch, avg_epoch_loss))
        if avg_epoch_loss < min_loss:
            min_loss = avg_epoch_loss
            torch.save(cnn.state_dict(), os.path.join(config.model_path, config.model_name))#保存avg_epoch_loss最小的model参数.'./model'文件夹下的model.pth文件

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR', choices=['STL', 'CIFAR', 'OWN'])
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--model_name', type=str, default='model.pth')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('-s', '--save_model_in_epoch', action='store_true')
    config = parser.parse_args()
    print(config)
    train(config)