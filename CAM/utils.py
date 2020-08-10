"""
下载数据集，并经过data.DataLoader弹出
"""
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def get_trainloader(dset_name, path, img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    if dset_name == "STL":
        dataset = dsets.STL10(root=path, split='train', transform=transform, download=True)
    elif dset_name == "CIFAR":
        dataset = dsets.CIFAR10(root=path, train=True, transform=transform, download=True)#获取数据集

    else:
        dataset = dsets.ImageFolder(root=path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset.classes)
           #返回dataloader供网络使用，同时返回dataset.classes=10


def get_testloader(dset_name, path, img_size, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    if dset_name == "STL":
        dataset = dsets.STL10(root=path, split='test', transform=transform, download=True)
    elif dset_name == "CIFAR":
        dataset = dsets.CIFAR10(root=path, train=False, transform=transform, download=True)
    else:
        dataset = dsets.ImageFolder(root=path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset.classes)


