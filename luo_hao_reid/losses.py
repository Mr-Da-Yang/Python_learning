'''
其中三元损失的难样本选择也在这里处理
'''
from __future__ import absolute_import
import sys

import torch
from torch import nn

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['DeepSupervision', 'CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss', 'RingLoss']

def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes#标签平滑技术
        loss = (- targets * log_probs).mean(0).sum()#sum(-ylog(y_hat))交叉商
        return loss

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):#三元组的阈值margin
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)#三元组损失函数
        #ap an margin y:倍率   Relu(ap - anxy + margin)这个relu就起到和0比较的作用

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)#32x2048
            targets: ground truth labels with shape (num_classes)#tensor([32])[1,1,1,1,2,3,2,,,,2]32个数，一个数代表ID的真实标签
        """
        n = inputs.size(0)#取出输入的batch
        # Compute pairwise distance, replace by the official when merged
        #计算距离矩阵，其实就是计算两个2048维之间的距离平方(a-b)**2=a^2+b^2-2ab
        #[1,2,3]*[1,2,3]=[1,4,9].sum()=14  点乘

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())#生成距离矩阵32x32，.t()表示转置
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability#clamp(min=1e-12)加这个防止矩阵中有0，对梯度下降不好
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())#利用target标签的expand，并eq，获得mask的范围，由0，1组成，，红色1表示是同一个人，绿色0表示不是同一个人
        dist_ap, dist_an = [], []#用来存放ap，an
        for i in range(n):#i表示行
            # dist[i][mask[i]],,i=0时，取mask的第一行，取距离矩阵的第一行，然后得到tensor([1.0000e-06, 1.0000e-06, 1.0000e-06, 1.0000e-06])
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))#取某一行中，红色区域的最大值，mask前4个是1，与dist相乘
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))#取某一行，绿色区域的最小值,加一个.unsqueeze(0)将其变成带有维度的tensor
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)#y是个权重，长度像dist-an
        loss = self.ranking_loss(dist_an, dist_ap, y) #ID损失：交叉商输入的是32xf f.shape=分类数,然后loss用于计算损失
                                                      #度量三元组：输入的是dist_an（从距离矩阵中，挑出一行（即一个ID）的最大距离），dist_ap
                                                     #ranking_loss输入 an ap margin y:倍率  loss： Relu(ap - anxy + margin)这个relu就起到和0比较的作用
        # from IPython import embed
        # embed()
        return loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

class RingLoss(nn.Module):
    """Ring loss.
    
    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """
    def __init__(self, weight_ring=1.):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))
        self.weight_ring = weight_ring

    def forward(self, x):
        l = ((x.norm(p=2, dim=1) - self.radius)**2).mean()
        return l * self.weight_ring

# if __name__ == '__main__':
#     target=[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8]
#     target=torch.Tensor(target)
#     features=torch.Tensor(32,2048)#生成一个32x2048维的0矩阵
#     a=TripletLoss()
#     a.forward(features,target)
