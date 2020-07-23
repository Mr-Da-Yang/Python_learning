#torchvision.datasets含有许多数据集
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

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

for data, label in train_loader:
    out=data.data.squeeze().numpy()
    plt.imshow(out,cmap='gray')
    # print(data.data.squeeze().shape)
    print('data:{} labe:{} out:{}'.format(data.shape, label.shape, out.shape))
    plt.show()
