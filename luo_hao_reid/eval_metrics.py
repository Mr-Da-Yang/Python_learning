from __future__ import print_function, absolute_import
import numpy as np
import copy
from collections import defaultdict
import sys

def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):#训练阶段用交叉熵直接进行反向传播，test阶段的query才需要距离矩阵
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape#距离矩阵，看博客的那张取难样本的图，行代表的是请求querry=3368，列表示的是galerry中的所有照片数15913
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)#返回的是distant中元素从小到大排序后，对应原来distant中元素所在位置的索引号
    #x=np.array([2,4,5,3,-10,1])，输出结果为：y=[4 5 0 3 1 2]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # 1.生成距离矩阵distmat(3368,15913)
    # 2.indices=np.argsort(distmat, axis=1)将距离矩阵从小到大排序，返回从小到大对应的索引(3368,15913)
    # 3.g_pids[indices]将上述得到的索引值，换成按索引值找寻到的图片标签，得到(3368,15913)
    # 4.matches将querry的(3368,1)与(3368,15913)相比较，求出0，1矩阵（3368,15913）

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):#num_q=3368
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]#indices=distant.sort()返回距离从小到大，照片的索引值，order选中一行
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)#去除同一个camera下的同一个ID,array([ True, False, False, ..., False, False, False])
        keep = np.invert(remove)#array([False,  True,  True, ...,  True,  True,  True])

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches # array([0, 0, 0, ..., 0, 0, 0]),(15905,)
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()#array([ 0,  1,  1, ..., 51, 51, 51], dtype=int32)),(15905,)
        cmc[cmc > 1] = 1 #array([ 0,  1,  1, ..., 1, 1, 1]元素大于1的都设为1，这样方便以后求rank，和cmc
        all_cmc.append(cmc[:max_rank])#array([0, 0, 1, 1, 1, 1, 1, ,,,,,1])一行取50个数据，max——rank=50
        num_valid_q += 1.# number of valid query

        # compute average precision mAP推导见博客
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        # a = [1, 2, 3, 4, 5, 6, 7]， np.cumsum(a)=array([  1,   3,   6,  10,  15,  21,  28])
        #orig_cmc,(15905,)
        num_rel = orig_cmc.sum()#51是一个数值
        tmp_cmc = orig_cmc.cumsum()#array([ 0,  1,  1, ..., 51, 51, 51], dtype=int32)
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)#all_cmc.shape=(3368,50)
    all_cmc = all_cmc.sum(0) / num_valid_q#对于all_cmc.shape=(3368,50)，进行第0维度的求和，得到1x50，然后再除以qerry数量，就得到rank1到rank50，共50列
                                          #all_cmc[0]:就是rank_1   all_cmc[1]:就是rank_2
    mAP = np.mean(all_AP)
    return all_cmc, mAP #这里all_cmc共50列，每一列就是一个rank，mAP是一个数值

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)