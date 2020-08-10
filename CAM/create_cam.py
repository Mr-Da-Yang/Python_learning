import os
import argparse
import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms

from CAM import utils
from CAM import model


def create_cam(config):
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)

    # 返回dataloader供网络使用，同时返回dataset.classes=10
    test_loader, num_class = utils.get_testloader(config.dataset,
                                        config.dataset_path,
                                        config.img_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = model.CNN(img_size=config.img_size, num_class=num_class).to(device)
    cnn.load_state_dict(
        torch.load(os.path.join(config.model_path, config.model_name))
    )#加载最优的model参数

    # hook信息的设置，只要图片数据一经过cnn，就会自动往feature_blobs = []中追加信息
    finalconv_name = 'conv'
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())
    cnn._modules.get(finalconv_name).register_forward_hook(hook_feature)#hook让hook_feature不释放


    # get weight only from the last layer(linear)
    params = list(cnn.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())#通过cnn.state_dict()来查看网络参数，最后两层是classifier.weight,classifier.bias
                                                              #weight_softmax.shape=(10,10)因为他是一个10分类任务
    """
    pars
    feature_conv: (1, 10, 16, 16)
    weight_softmax, (10,10)
    class_idx=idx[0].item()  idx=(1,10),   idx[0].item()返回最大预测概率最大的类
    """
    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (config.img_size, config.img_size)#128x128

        _, nc, h, w = feature_conv.shape#(1, 10, 16, 16)
        output_cam = []
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))#weight_softmax[class_idx]返回预测概率最大类相应的权重，但是预测不一样准确
       #cam(1, 256)

       #cam规一化
        cam = cam.reshape(h, w)#(16, 16)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)#0-1之间的值
        cam_img = np.uint8(255 * cam_img)#0-255之间的值#(16, 16)
        output_cam.append(cv2.resize(cam_img, size_upsample))#把(16,16)扩到（128,128）目的为了与原来的(128，128)图片重叠放置
        return output_cam

    """
    tensor读的图片数据是3,128,128,   cv2.imread读的是128,128,3
    """
    class_name=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i, (image_tensor, label) in enumerate(test_loader):#image_tensor.shape,torch.Size([1, 3, 128, 128])
        from IPython import embed
        image_PIL = transforms.ToPILImage()(image_tensor[0])#取出第一张图片image_tensor[0].shape=torch.Size([3, 128, 128]),type(image_PIL)=PIL.Image.Image
        image_PIL.save(os.path.join(config.result_path, 'img%d.png' % (i + 1)))
        image_tensor = image_tensor.to(device)
        #feature_blobs=[]
        logit, _ = cnn(image_tensor)#logit.shape=torch.Size([1, 10]),表示一张图片预测的10个数
        #因为前面已经设置好要hook的位置，所以只要数据一经过cnn训练，feature_blobs里面就会存有数据feature_blobs[0].shape=(1, 10, 16, 16)
        h_x = F.softmax(logit, dim=1).data.squeeze()#归一化上面有负有正的值
        probs, idx = h_x.sort(0, True)#按从小到大顺序排列P，和相应的序号
        print("True label : %d, name : %s \nPredicted label : %d, name : %s Predicted_probability : %.2f" % (
            label.item(),class_name[label.item()], idx[0].item(),class_name[idx[0].item()], probs[0].item()))
        #True label : 7, Predicted label : 4, Probability : 0.98,拿到一张照片，通过softmax分析后，发现有98%的概率是4号类别，但真实类别为7，所以判断错了
        CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])#根据前面的设置，feature_blobs[]会自动追加数据,CAMs返回.shape=(128,128)的CAM图
        img = cv2.imread(os.path.join(config.result_path, 'img%d.png' % (i + 1)))#(128, 128, 3)
        height, width, _ = img.shape#(128, 128, 3)
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)#因为CAMs是一个list，CAMs[0].shape=(128,128),heatmap.shape=(128, 128, 3)
        result = heatmap * 0.3 + img * 0.5#heatmap和原图叠加
        cv2.imwrite(os.path.join(config.result_path, 'cam%d.png' % (i + 1)), result)
        cv2.imwrite(os.path.join(config.result_path, 'heatmap%d.png' % (i + 1)), heatmap)
        if i + 1 == config.num_result:#设置只保存一张图片的heatmap
            break
        feature_blobs.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR', choices=['STL', 'CIFAR', 'OWN'])
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--model_name', type=str, default='model.pth')

    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_result', type=int, default=1)

    config = parser.parse_args()
    print(config)

    create_cam(config)