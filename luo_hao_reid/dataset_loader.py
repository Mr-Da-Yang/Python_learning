"""OK
1.data_manager.init_img_dataset调用class Market1501(object)返回的结果是train=(path，poseid，cameraid,)
2.dataset_loader.class ImageDataset(Dataset)，将上面的地址变成图片     return img, pid, camid
"""
#data_manager后，用PIL对图片的读取
from __future__ import print_function, absolute_import
import matplotlib.pyplot as plt
import os
from PIL import Image#读图片用
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):#path是否存在，不存在就raise
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True#当为True时，循环跳出
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return  img

class ImageDataset(Dataset):#torch自带的dataset可能不满足需求，自己重构,这里读取地址，获得图片
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):#transform对数据处理
        self.dataset = dataset
        self.transform = transform#变成类属性，供调用
        print(len(self.dataset))#self.dataset是一个列表，里面每个元素包含（照片位置，poseid，cameraid）
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]#取出dataset元祖的数据
        # print("*"*50,img_path)#img_path是self.dataset是列表中的一个个元素
        img = read_image(img_path)#img_path是dataset[index]的图片地址，img是图片
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

# if __name__ =="__main__":
#     import data_manager
#     #使用data_manager.init_img_dataset，得到一个类，实例化后，给上数据的path，获取数据
#     dataset = data_manager.init_img_dataset(root='E:\LINYANG\python_project\luohao_person_reid\dataset',name='market1501')
#
#     train_loader = ImageDataset(dataset.train)#dataset是一个对象，取dataset.trian属性，获得trian数据集是一个列表
#     for batch_id,(img,pid,camid) in enumerate(train_loader):
#         print(batch_id,(img,pid,camid))
#         # plt.imshow(img)
#         # plt.show()
#     from IPython import embed
#     embed()

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order
            # comment it to be order-agnostic
            indices = np.sort(indices)
        elif self.sample == 'evenly':
            """Evenly sample seq_len items from num items."""
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num/self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid