"""
创建自己的cnn网络
"""
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, img_size, num_class):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            # 3 x ? x ?
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # 32 x ? x ?
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # 64 x ? x ?
            nn.MaxPool2d(2, 2),
            # 64 x ? / 2 x ? / 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # 128 x ? / 2 x ? / 2
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # 256 x ? / 2 x ? / 2
            nn.MaxPool2d(2, 2),
            # 256 x ? / 4 x ? / 4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # 512 x ? / 4 x ? / 4
            nn.MaxPool2d(2, 2),
            # 512 x ? / 8 x ? / 8
            nn.Conv2d(512, num_class, 3, 1, 1),#这里使用CIFAR10数据集，最后输出维度num_class=10个通道，保证10个权重分别分配到每个通道上
            nn.LeakyReLU(0.2)
        )
        self.avg_pool = nn.AvgPool2d(img_size // 8)
        self.classifier = nn.Linear(num_class, num_class)

    def forward(self, x):
        features = self.conv(x)
        flatten = self.avg_pool(features).view(features.size(0), -1)
        output = self.classifier(flatten)
        return output, features
