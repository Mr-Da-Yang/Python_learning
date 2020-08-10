'''OK
transform实现数据增广，pytorch中自带的随机翻转，剪切可以解决90%的问题，下面是自己如何做一个transform
'''
from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
class Random2DTranslation(object):#随机放大并crop
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self,  height,width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation#放大时用的插值

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:#random产生一个0-1分步的随机数，若数字小于P，就不进行数据增广
            return img.resize((self.width, self.height), self.interpolation)
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))#放大1/8/round取上限，加上int更稳
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height#crop的最大范围
        x1 = int(round(random.uniform(0, x_maxrange)))#crop的起始点
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

# if __name__ == '__main__':
#     import torchvision.transforms as transform
#     img=Image.open('E:\LINYANG\python_project\luohao_person_reid\dataset\Market-1501-v15.09.15\query\\0004_c1s6_016996_00.jpg')
#
#     transform=transform.Compose([
#         Random2DTranslation(128, 256, 0.5),
#         transforms.RandomHorizontalFlip(p=0.8),
#         #transform.ToTensor()
#         ])
#     img_t=transform(img)
#     plt.subplot(121)
#     plt.imshow(img)
#     plt.subplot(122)
#     plt.imshow(img_t)
#     plt.show()
