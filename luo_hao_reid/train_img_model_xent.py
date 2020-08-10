'''OK
Train image model with cross entropy loss
'''
from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from luohao_person_reid import data_manager
from luohao_person_reid.dataset_loader import ImageDataset
from luohao_person_reid import transforms as T
from luohao_person_reid import models
from luohao_person_reid.losses import CrossEntropyLabelSmooth, DeepSupervision
from luohao_person_reid.utils import AverageMeter, Logger, save_checkpoint
from luohao_person_reid.eval_metrics import evaluate
from luohao_person_reid.optimizers import init_optim

parser = argparse.ArgumentParser(
    description='Train image model with cross entropy loss')  # argparse.ArgumentParser()获取所有参数，args = parser.parse_args()#解析参数
# Datasets
parser.add_argument('--root', type=str, default='E:\LINYANG\python_project\luohao_person_reid\dataset', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")  # ,'--workers'参数名，'-j'，参数名的简写，workers读取数据的线程
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")  # 图片的长乘宽256*128
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")  # split_id 针对CHK03数据集，这里没有

# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")

# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")  # 从哪个epoch开始训练，中间可以中断
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=20, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")  # 多少个epoch学习率降一次
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")  # 学习率每次降当前的5e-04
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")  # 为防止过拟合，加了正则项，正则项的加入，模型的复杂度会成为loss的一部分
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())  # 选择model
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")  # 多少频率打印loss
parser.add_argument('--seed', type=int, default=1, help="manual seed")  # 控制随机参数，保征每次结果稳定复现
parser.add_argument('--resume', type=str, default='', metavar='PATH')  # 读档，读某个路径的
parser.add_argument('--evaluate', action='store_true', help="evaluation only")  # 因为测试和训练在一个py，所以通过其来控制，是测试还是训练
parser.add_argument('--eval-step', type=int, default=-1,  # 之前为-1
                    help="run evaluation for every N epochs (set to -1 to test after training)")  # 每隔多少个epoch进行一次测试，默认-1，说明训练完再测试
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")  # 多少个epoch后进行测试
parser.add_argument('--save-dir', type=str, default='log')  # 存放loss的路径
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()  #


def main():
    '''以下作用1.存储损失 2.gpu，cpu的选择'''
    torch.manual_seed(args.seed)  # args.seed默认为1，保证每次结果可以复现
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    if not args.evaluate:  # 训练模式，损失存在log_train.txt中，测试模式存在log_test.txt
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))  # 存放log文件
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))  # 打印所有arg中使用的参数

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))  # 打印使用的GPU
        cudnn.benchmark = True  # 使用gpu
        torch.cuda.manual_seed_all(args.seed)  # 传入随机数的总数，牵扯到随机初始化，加上这个保证结果稳定复现
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # 1.制定数据集 2.制定transform 3.制作trainloader吐出tensor进行训练
    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )  # args.root的根目录，name数据集，split_id 针对CHK03数据集，这里没有

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),  # 只有train时，才用数据增广
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # 将图片格式转化为电脑识别出来的tensor格式
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 图像像素值的归一化，方便计算，约定俗成的常数
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),  # 因为test数据有大有小，所以要resize成一样大
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False
    # 以下为三个dataloader，其中trainloader需要数据增广，其他不用
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),  # transform有增广，
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers,  # num_workers读数据的现成
        pin_memory=pin_memory, drop_last=True,
        # pin_memory节省内存模式打开 drop_last：一批有100张图片，batch=32，so 一共分成3组，剩下的4张就丢弃，否则batch不一样大会报错
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,  # shuffle不用打乱
        pin_memory=pin_memory, drop_last=False,  # test时，每张图都保留
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    # model的选择，通过models.init_model选择resnet类对象，并给参数
    print("Initializing model: {}".format(args.arch))  # args.arch用来选择resnet50框架
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent'},
                              use_gpu=use_gpu)  # dataset.num_train_pids多少个pid多少个分类
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # 1.loss 2.optimizer 3.学习率的控制 4.resume重载数据 5.model包装成并行，放到多卡上训练
    criterion = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    optimizer = init_optim(args.optim, model.parameters(), args.lr,
                           args.weight_decay)  # model.parameters()更新模型的所有参数，weight_decay正则
    # #用nn.Sequential来包裹model的两层参数，每次更新只更新这两层
    # optimizer = init_optim(args.optim, nn.Sequential([model.conv1,model.conv2]), args.lr, args.weight_decay)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize,
                                        gamma=args.gamma)  # args.stepsize多少个epoch学习率降一次学习率阶梯式衰减，args.gamma衰减倍数
    start_epoch = args.start_epoch  # 从第几个epoch训练，中间你干别的事可以暂停训练

    if args.resume:  # 训练的读档，根据之前产生的数据，加载到模型中，恢复模型
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)  # 读路径下的参数
        model.load_state_dict(checkpoint['state_dict'])  # 1.将权重的参数加载到模型中
        start_epoch = checkpoint['epoch']  # 2. 第几个epoch的参数

    if use_gpu:
        model = nn.DataParallel(model).cuda()  # 将model包装成并行，放到多卡上训练
        # model.module.parameters()#多卡的parameters

    if args.evaluate:  # 如果只是测试的话
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0

    ##
    print("==> Start training")
    for epoch in range(start_epoch, args.max_epoch):  # 从起始的epoch到最大epoch
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)  # 下面train函数的使用
        train_time += round(time.time() - start_train_time)

        if args.stepsize > 0: scheduler.step()  # 加了这句学习率都会衰减

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            # 用checkpint函数封装train的数据
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))  # 训练后的数据存到这个路径

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


# 训练函数，并输出损失
def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    losses = AverageMeter()  # 调用utils的函数，计算平均、总值使用
    batch_time = AverageMeter()  # 前传加返传时间
    data_time = AverageMeter()  # measure data loading time

    model.train()  # model处于train模式

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(
            trainloader):  # trainloader吐出tensor数据，之前是imgs, pids, cameraid 这里cameraid用不到
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()  # 如果使用GPU模式，图片要放到cuda下，cuda两步骤1.模型放上去2.每次循环把输入数据放上去

        # measure data loading time
        data_time.update(time.time() - end)  # 主函数有个起始时间time.time()，读数据时间

        outputs = model(imgs)  # 输出的分类标签torch.Size([32, 751])

        if isinstance(outputs, tuple):
            loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)  # 12936的每张输入对应pids的750人的一个标签，这里就是交叉商损失。可参考mnist,根据pytorch的官方文档，size_average默认情况下是True，对每个小批次的损失取平均值。
        optimizer.zero_grad()
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新网络参数

        # measure elapsed time
        batch_time.update(time.time() - end)  # 前传加返传时间
        end = time.time()

        losses.update(loss.item(), pids.size(0))  # loss.item(), pids.size(0)输出(6.74575138092041, 32)
        if (batch_idx + 1) % args.print_freq == 0:
            # Epoch: [1][10/404]	Time 0.145 (1.174)	Data 0.024 (0.589)	Loss 6.8756 (6.8195)
            # Epoch: [1][20/404]	Time 0.143 (0.658)	Data 0.021 (0.303)	Loss 6.7289 (6.8909)
            # Epoch: [1][30/404]	Time 0.136 (0.485)	Data 0.015 (0.209)	Loss 6.5819 (6.8173)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                # trainloader长度为404，每个trainloader里数据为32
                data_time=data_time, loss=losses))


# 测试阶段，
def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()  # model的测试阶段,  如果不加.eval(),下面人的features = model(imgs)=torch.Size([32, 751])

    # querry共有3368人，每个人提取2048维特征，qf.shape=torch.Size([3368, 2048])
    with torch.no_grad():  # 不用求导
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            # len(queryloader)=106,x32=3392,实际3368人
            # imgs.Size([32, 3, 256, 128])
            # pids.torch.Size([32]),tensor([1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6,6, 6, 6, 8, 8, 8, 9, 9])
            if use_gpu: imgs = imgs.cuda()
            end = time.time()
            features = model(imgs)  # torch.Size([32, 2048])用于距离矩阵返回querry，对于train的model输出来讲是torch.Size([32, 2048])

            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)  # torch.Size([3368, 2048])
        q_pids = np.asarray(q_pids)  # Out[4]: (3368,)
        q_camids = np.asarray(q_camids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        # 提取gallery中15913个人，每个人生成2048为向量，gf.shape=torch.Size([15913, 2048])
        gf, g_pids, g_camids = [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):  # 498x32=15936；实际排除-1标签后人数据为15913
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)  # torch.Size([32, 2048])
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)

        gf = torch.cat(gf, 0)  # torch.Size([15913, 2048])
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")  # 这个cmc就是rank_1--rank_n
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)
    # distmat.shape=(3368, 15913) q_pids.shape=(3368,) g_pids.shape=(15913,) q_camids=(3368,) g_camids=(15913,)
    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc[0]


if __name__ == '__main__':
    main()