#每个行人取4张图片，生成数据。用于后面losses.py难样本三元组损失，挑选A,P,N，
from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):#继承torch的sampler类，里面有三个函数，所以自己也写这三个类
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to  sample from.
        num_instances (int): number of instances per identity.
    """

    # from collections import defaultdict#创建字典，方便存储同一ID的多张照片，自动增长迭代器
    # a = defaultdict(list)
    # print(a)
    # a[0].append('b')
    # a[1].append('c')
    # a[1].append('d')
    # print(a)        defaultdict(<class 'list'>, {0: ['b'], 1: ['c', 'd']})
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        #下面3行是个字典，方便以后采样
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)#获得一个字典，{0:[0,1,2,3,4,5],...750:[12928,12929,12930]}#key=750,一个key中的照片数量不等

        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)
        # from IPython import embed#加断点查看内容
        # embed()
##iter迭代器，返回一个list. 751x4=3004， batch=32 每个ID4张图，[0,12,3,5,78,468,5,6''']连续4张0，12，3，5表示第一个ID,一个batch取32张图，相当于8个ID
    def __iter__(self):#迭代器，返回一个list
        indices = torch.randperm(self.num_identities)#self.num_identities是一个数字750，生成的indices是一维数组tensor([750]),里面0-750不同顺序
        ret = []
        for i in indices:#indices是一个打乱顺序的列表,这里为啥要打乱，是因为batch是要进行shuffle的，但是这样的会乱，还不如直接生成的ret=751x4，这里每次751个ID顺序不同，ret只要保证前后的4个数是一个人
            pid = self.pids[i]#pid就是从0-750中随机挑选一个数字
            t = self.index_dic[pid]#选中某一个ID，得到他的多张图片
            replace = False if len(t) >= self.num_instances else True#repalce用于控制提出的4张图片是否有重复，如果一个ID只有2张图片，你偏要提取4张，就会报错
            t = np.random.choice(t, size=self.num_instances, replace=replace)#从一个ID的多张图片，选择self.num_instances张，用于P*K的难样本挑选
            ret.extend(t)
        # from IPython import embed
        # embed()
        return iter(ret)#ret是个751x4=3004列表,0-3为一个ID，4-7为一个ID，这样3004个图，每个batch=32，3004/32=93.87丢弃几个，所以1个epoch3004张图，迭代93次
                        #这样 就不用shuffle
    def __len__(self):#迭代器属性问题
        return self.num_identities * self.num_instances#751x4=3004一个epoch长度

# if __name__=='__main__':
#     from luohao_person_reid.data_manager import Market1501
#     dataset = Market1501(root='E:\LINYANG\python_project\luohao_person_reid\dataset')
#     sampler = RandomIdentitySampler(dataset.train,num_instances=4)
#     a=sampler.__iter__()
