#ut.shape=(28, 28) type(out)=torch.Tensor:二维的out_tensor可直接用于plt
#torchvision.datasets含有许多数据集
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import os
import time

batch_size = 1
train_dataset = datasets.MNIST(root='E:\LINYANG\python_project\learn_pytorch\dataset',
                                train=True,
                                download=True,
                                transform=transforms.ToTensor(),
                                 )
train_loader = DataLoader(train_dataset,
                          shuffle=False,
                          batch_size=batch_size
                          )

# for data, label in train_loader:#经过dataloader后变成Tensor格式，'Tensor' object has no attribute 'Image.read'
#     out=data.squeeze()#out.shape=(28, 28) type(out)=torch.Tensor ,data.shape=torch.Size([1, 1, 28, 28])
#     plt.imshow(out,cmap='gray')#Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r
#     # print(data.data.squeeze().shape)
#     print('data:{} labe:{} out:{}'.format(data.shape, label.shape, out.shape))
#     plt.pause(1)

for i,(data, label) in enumerate(train_loader):#经过dataloader后变成Tensor格式，'Tensor' object has no attribute 'Image.read'
    #data.shape=torch.Size([1, 1, 28, 28])
    image_PIL = transforms.ToPILImage()(data[0])#type(image_PIL)=PIL.Image.Image
    plt.imshow(image_PIL)
    plt.pause(1)
    image_PIL.save(os.path.join('./save_mnist_picture', 'img%d.png' % (i + 1)))#save的图是黑白的，plt是彩色的
