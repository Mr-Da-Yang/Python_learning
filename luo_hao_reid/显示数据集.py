#读文件夹中的照片
import os.path as osp
import glob
import matplotlib.pyplot as plt
from PIL import Image#读图片用
from torchvision import datasets, models, transforms

# print(type(img_paths))
# print(img_paths[0])
#Image.open(path)中的path只能读一个具体的文件不能读文件夹
path0='E:\LINYANG\python_project\luohao_person_reid\dataset\Market-1501-v15.09.15\\bounding_box_test\\0000_c1s1_000151_01.jpg'#YES
path1='E:\LINYANG\python_project\luohao_person_reid\dataset\Market-1501-v15.09.15\\bounding_box_test'#NO
path2 = glob.glob(osp.join('E:\LINYANG\python_project\luohao_person_reid\dataset\Market-1501-v15.09.15\\bounding_box_test', '*.jpg'))#拿出path3下所有后缀带.jpg的文件，并给出绝对路径

for i in range(5):
    img = Image.open(path2[i]).convert('RGB')#type(path2)是list
    # plt.figure(i)#每次创建一个画图(不放这句的话图形显示连贯)，可以在最后加一个plt.close()关闭每个figure
    plt.imshow(img)
    plt.pause(0.2)#没有这句话，图片显示太快看不见
plt.show()#这是最终的一个显示，加上这句，图片就不连着放，需要鼠标点

#生成两幅图片，同时显示两个画布
# img = Image.open(path2[0]).convert('RGB')
# img1 = Image.open(path2[1]).convert('RGB')
# plt.figure(1)
# plt.imshow(img)
# plt.figure(2)
# plt.imshow(img1)
# plt.show()




