import os
import sys
import math
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from network.network_u2pl import my_net
from utils.utils import get_device, check_accuracy, check_accuracy_dual, label_to_onehot, dequeue_and_enqueue
from mms_dataloader_dy_u2pl import get_meta_split_data_loaders
from config_u2pl_dy import default_config
from utils.dice_loss import dice_coeff
import utils.mask_gen as mask_gen
from utils.custom_collate import SegCollate

# multiple GPU setting
# gpus = default_config['gpus']
# torch.cuda.set_device('cuda:{}'.format(gpus[0]))

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
use_cuda = torch.cuda.is_available()

wandb.init(project='MNMS_SemiDG_U2PL_DY', entity='du-yao',
           config=default_config, name=default_config['train_name'])
config = wandb.config
# device = get_device()


# 分布式计算 torch.distributed
# 数据并行 torch.nn.DataParallel
# torch.distributed 在调用前，
# torch.distributed.init_process_group('nccl',init_method='env://',world_size=1,rank=0)

# ------------------------------point 9 ------------------------------

# 这里对pre_data做了修改，对于label data, 其不再进行fourier aug,保证其本身的变化不会太大
# 当然，这里对于label data,做了fourier aug,也可以根据KL散度，对损失函数进行调整，综合比较最终的优劣。


def pre_data(batch_size, num_workers, test_vendor):

    test_vendor = test_vendor

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
        domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
        test_dataset = get_meta_split_data_loaders(
            test_vendor=test_vendor, image_size=224)

    # val dataset is the total labeled data
    val_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    label_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    unlabel_dataset = ConcatDataset(
        [domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset])

    print("before length of label_dataset", len(label_dataset))
    new_labeldata_num = len(unlabel_dataset) // len(label_dataset) + 1

    # point 1----------

    # match the dimension between labeled and unlabeled dataset for mix.
    # this operation is similar for mix-match
    # 这里对于labeled dataset, 简单的进行了扩充，使labeled 和 unlabeled 的数量进行匹配

    new_label_dataset = label_dataset
    for i in range(new_labeldata_num):
        new_label_dataset = ConcatDataset([new_label_dataset, label_dataset])
    label_dataset = new_label_dataset

    # For CutMix
    mask_generator = mask_gen.BoxMaskGenerator(prop_range=config['cutmix_mask_prop_range'],
                                               n_boxes=config['cutmix_boxmask_n_boxes'],
                                               random_aspect_ratio=config['cutmix_boxmask_fixed_aspect_ratio'],
                                               prop_by_area=config['cutmix_boxmask_by_size'],
                                               within_bounds=config['cutmix_boxmask_outside_bounds'],
                                               invert=config['cutmix_boxmask_no_invert'])

    add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
        mask_generator
    )
    # collate与字符类型的排序有关
    collate_fn = SegCollate()
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

    # Dataloader generation
    label_loader = DataLoader(dataset=label_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    # two individual unlabeled data generator
    unlabel_loader_0 = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, drop_last=True, pin_memory=False, collate_fn=mask_collate_fn)

    unlabel_loader_1 = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    # 80848, 80352, 248, 1266
    print("after length of label_dataset", len(label_dataset))
    print("length of unlabel_dataset", len(unlabel_dataset))
    print("length of val_dataset", len(val_dataset))
    print("length of test_dataset", len(test_dataset))

    return label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, len(label_dataset), \
        len(unlabel_dataset)


def dice_loss(pred, target):

    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 0.1  # 1e-12

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    loss = ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth)).mean()

    return 1 - loss


def total_dice_loss(pred, target):

    # ---------point2

    # 这里要注意下样本真实的label标签的顺序（0,1,2,3）的代表顺序，0是bg还是lv

    # the output is a 4d tensor (batch=size, class dimension, H, W)
    dice_loss_lv = dice_loss(pred[:, 0, :, :], target[:, 0, :, :])
    dice_loss_myo = dice_loss(pred[:, 1, :, :], target[:, 1, :, :])
    dice_loss_rv = dice_loss(pred[:, 2, :, :], target[:, 2, :, :])
    dice_loss_bg = dice_loss(pred[:, 3, :, :], target[:, 3, :, :])

    loss = dice_loss_lv + dice_loss_myo + dice_loss_rv + dice_loss_bg

    return loss


# -------------------------------point3-------------------------------

# 这里将并行的两个模型改为EMA model，减小一半的模型更新计算量.这样只需要初始化一个model即可


def ini_model_dy():

    model = create_model()
    ema_model = create_model(ema=True)

    # model = model.to(device)
    # model.device = device
    #
    # ema_model = ema_model.to(device)
    # ema_model.device = device

    model = model.cuda()
    ema_model = ema_model.cuda()

    # 数据并行
    # model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    # ema_model = nn.DataParallel(ema_model, device_ids=gpus, output_device=gpus[0])

    return model, ema_model


def create_model(ema=False, restore=False, restore_from=None):

    if restore:
        model_path = './tmodel/u2pl/' + str(restore_from)
        model = torch.load(model_path)
        print("restore from", model_path)
    else:
        model = my_net(modelname='mydeeplabV3P')

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


class WeightEMA(object):

    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * default_config['learning_rate']

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                param.mul_(1 - self.wd)


class SquareLoss(object):

    def __call__(self, outputs_u, targets_u):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lu


def ini_optimizer_dy(model, ema_model, learning_rate, weight_decay,ema_decay):

    # Initialize two optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # step_schedule = torch.optim.lr_scheduler.StepLR(step_size=10, gamma=0.9, optimizer=optimizer)
    step_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=10)

    ema_optimizer = WeightEMA(model, ema_model, alpha=ema_decay)

    return optimizer, ema_optimizer, step_schedule


def compute_unsupervised_loss(predict, target, percent, pred_teacher):

    # 这里percent的意义在于表明，对于通过伪标签得到的无标签数据，其中后20%的比例，认为其伪标签是不可靠的，不应该用来计算监督学习损失。

    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():

        # drop pixels with high entropy, 转化到0,1
        # prob size: batch * num_class * h * w
        prob = torch.softmax(pred_teacher, dim=1)
        # 这里的entropy衡量最后是正数，所以熵小的，优质样本在前面，熵大的在后边
        # entropy size: batch * h * w
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        # 提取到pixel中前80%的中位数
        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )

        # 经过这一步，将unreliable pixel对应的像素位置提取出来
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()
        # 所以这里255的意思就是将unreliable对应的pixel的标签赋值为255
        # 这些pixel将不参与伪标签监督学习对应的loss

        target[thresh_mask] = 255

        # 这里计算的是剩下的参与监督学习的label的数量对应的百分率倒数。
        weight = batch_size * h * w / torch.sum(target != 255)

    # 有监督学习，对应的伪标签数据loss计算，将unreliable　pixel筛除出去
    loss = weight * F.cross_entropy(predict, target, ignore_index=255)  # [10, 321, 321]

    return loss


def compute_contra_memobank_loss(rep,label_l,label_u,prob_l,prob_u,low_mask,high_mask,memobank,queue_prtlis,
                                 queue_size,rep_teacher,momentum_prototype=None,i_iter=0):

    # 筛选anchor　pixel, 要求对应的模型预测概率大于阈值(0.3)
    current_class_threshold = default_config['current_class_threshold']
    current_class_negative_threshold = default_config['current_class_negative_threshold']
    # 上下限　对pixel中相应的label 和　unlabeled pixel进行提取，进行对比学习
    low_rank, high_rank = default_config['low_rank'], default_config['high_rank']
    temp = default_config['temperature']
    # 单个anchor进行对比学习需要的num_negative_sample
    num_queries = default_config['num_queries']
    # 每个class对应的anchor数量
    num_negatives = default_config['num_negatives']
    # number of feature dim
    num_feat = rep.shape[1]
    # print(num_feat)
    # batch size
    num_labeled = label_l.shape[0]
    # number of channel/number of class
    num_segments = label_l.shape[1]

    # 对于不稳定的无标签数据，其本身的伪label已经被赋值为255(自动屏蔽)
    # 但是其对应的feature和prob仍然有用，可以用来指导对比学习训练
    # 前80%的较高质量无标签pixel,已经放入有监督学习中，计算相应的交叉熵loss
    # 对于后20%的pixel,则用于无监督对比学习。low_pixel用来计算anchor

    # one-hot like label
    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
    high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

    # print(rep.size())
    rep = rep.permute(0, 2, 3, 1)
    rep_teacher = rep_teacher.permute(0, 2, 3, 1)

    seg_feat_all_list = []
    seg_feat_low_entropy_list = []  # candidate anchor pixels
    seg_num_list = []  # the number of low_valid pixels in each class
    seg_proto_list = []  # the center of each class (positive sample)

    # 通道数为１代表在class prob这个维度上进行降序; 并将降序后的通道序列表示出来
    _, prob_indices_l = torch.sort(prob_l, 1, True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)  # (num_unlabeled, h, w, num_cls)

    # the dim of one-hot label is the same as that of prob now
    prob = torch.cat((prob_l, prob_u), dim=0)  # (batch_size, num_cls, h, w)

    valid_classes = []
    new_keys = []

    # 按照class分别计算其对比损失
    for i in range(num_segments):

        low_valid_pixel_seg = low_valid_pixel[:, i]

        high_valid_pixel_seg = high_valid_pixel[:, i]

        prob_seg = prob[:, i, :, :]

        rep_mask_low_entropy = (prob_seg > current_class_threshold) * low_valid_pixel_seg.bool()

        rep_mask_high_entropy = (prob_seg < current_class_negative_threshold) * high_valid_pixel_seg.bool()

        # extract anchor pixel features
        seg_feat_all_list.append(rep[low_valid_pixel_seg.bool()])

        seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])

        # extract positive anchor features (mean of anchor features)
        seg_proto_list.append(
            torch.mean(
                rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
            )
        )

        # torch.eq()函数逐元素进行比较，若相同，则返回true;反之，为false.
        # 将该次预测针对当前class,其对应预测概率值在low/high之间的对应像素点位置提取出来
        class_mask_u = torch.sum(prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3).bool()

        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

        # extract unreliable labeled and unlabeled pixels for contrastive learning
        class_mask = torch.cat((class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0)

        negative_mask = rep_mask_high_entropy * class_mask

        # 当前具体参与对比学习训练的负样本(包含了有标签和无标签的数据）
        keys = rep_teacher[negative_mask].detach()

        # keys中负样本的数量(有效的负pixel的数量）
        new_keys.append(
            dequeue_and_enqueue(
                keys=keys,
                queue=memobank[i],
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],
            )
        )

        if low_valid_pixel_seg.sum() > 0:
            seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes.append(i)

    # 如果说当前batch中找不到有效的anchor pixel，则这一步不计算相应的对比损失
    if len(seg_num_list) <= 1:
        # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        if momentum_prototype is None:
            return new_keys, torch.tensor(0.0) * rep.sum()
        else:
            return momentum_prototype, new_keys, torch.tensor(0.0) * rep.sum()

    else:
        reco_loss = torch.tensor(0.0).cuda()

        seg_proto = torch.cat(seg_proto_list)
        # shape: [valid_seg, 256]

        valid_seg = len(seg_num_list)
        # number of valid classes

        # 所有类的所有negative sample feature的集合大小
        prototype = torch.zeros((prob_indices_l.shape[-1], num_queries, 1, num_feat)).cuda()

        for i in range(valid_seg):
            if (
                len(seg_feat_low_entropy_list[i]) > 0
                and memobank[valid_classes[i]][0].shape[0] > 0
            ):
                # select anchor pixel
                # 产生对应范围的一组随机数（在这里是256维）
                seg_low_entropy_idx = torch.randint(
                    len(seg_feat_low_entropy_list[i]), size=(num_queries,)
                )
                # 从feature list中按照产生的随机数组挑选出本次batch训练的anchor
                anchor_feat = (
                    seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda()
                )
            else:
                # in some rare cases, all queries in the current query class are easy
                reco_loss = reco_loss + 0 * rep.sum()
                continue

            # apply negative key sampling from memory bank (with no gradients)

            with torch.no_grad():

                negative_feat = memobank[valid_classes[i]][0].clone().cuda()

                # 长度是num_anchor * num_negative　对所有的都遍历进行相应的计算

                high_entropy_idx = torch.randint(
                    len(negative_feat), size=(num_queries * num_negatives,)
                )

                negative_feat = negative_feat[high_entropy_idx]

                negative_feat = negative_feat.reshape(
                    num_queries, num_negatives, num_feat
                )

                positive_feat = (
                    seg_proto[i]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(num_queries, 1, 1)
                    .cuda()
                )  # (num_queries, 1, num_feat)

                # 对每个类对应的positive feature进行EMA更新
                if momentum_prototype is not None:
                    if not (momentum_prototype == 0).all():
                        ema_decay = min(1 - 1 / i_iter, 0.999)
                        positive_feat = (
                            1 - ema_decay
                        ) * positive_feat + ema_decay * momentum_prototype[
                            valid_classes[i]
                        ]

                    prototype[valid_classes[i]] = positive_feat.clone()

                all_feat = torch.cat((positive_feat, negative_feat), dim=1)
                # (num_queries, 1 + num_negative, num_feat)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)

            # -----------20220428----------

            # 计算CE Loss,将对应所有的label项都赋值为０，最后结果不会是０吗？？？

            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())

        if momentum_prototype is None:
            return new_keys, reco_loss / valid_seg
        else:
            return prototype, new_keys, reco_loss / valid_seg


def linear_rampup(current, rampup_length=config['num_epoch']):

    if rampup_length == 0:
        return 1.0
    else:
        # np.clip将对应数组中的元素限制在参数所列出的最大最小值之间,当超出这个范围,将超出的值自动替换为对应的最大最小值.
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def cal_variance(pred, aug_pred):

    kl_distance = nn.KLDivLoss(reduction='none')
    # dimension (batch, class, height, width)
    sm = torch.nn.Softmax(dim=1)
    log_sm = torch.nn.LogSoftmax(dim=1)
    # 这里将所有batch 所有pixel的对应的散度都相加
    variance = torch.sum(kl_distance(
        log_sm(pred), sm(aug_pred)), dim=1)
    exp_variance = torch.exp(-variance)

    return variance, exp_variance


def label_onehot(inputs, num_segments):

    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)


def train_one_epoch_dy(model, niters_per_epoch, label_dataloader, unlabel_dataloader_0, unlabel_dataloader_1, optimizer,
                       ema_optimizer, step_schedule, cross_criterion, epoch, memobank, queue_ptrlis, queue_size):

    global prototype

    # loss data
    total_loss = []
    total_loss_sup = []
    total_unsup_loss = []
    total_con_loss = []

    # tqdm
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(niters_per_epoch),
                file=sys.stdout, bar_format=bar_format)

    # 每个batch进行更新

    for idx in pbar:

        i_iter = epoch * len(label_dataloader) + idx
        minibatch = label_dataloader.next()
        unsup_minibatch_0 = unlabel_dataloader_0.next()
        unsup_minibatch_1 = unlabel_dataloader_1.next()

        # 这里的fourier augmentation已经是一个非常强的变换了
        # 可以考虑在第二次实验中，是否对label，unlabel统一做一次mixup,而不是只针对unlabeled data.

        imgs = minibatch['img']
        aug_imgs = minibatch['aug_img']
        mask = minibatch['mask']
        # print(mask[0][2])

        unsup_imgs_0 = unsup_minibatch_0['img']
        unsup_imgs_1 = unsup_minibatch_1['img']

        aug_unsup_imgs_0 = unsup_minibatch_0['aug_img']
        aug_unsup_imgs_1 = unsup_minibatch_1['aug_img']

        mask_params = unsup_minibatch_0['mask_params']

        # --------------------point8--------------------

        # 在对label data 施加傅里叶强变换后，会对本身的label产生较大的影响
        # 对经过fourier aug img, 其原有标签是否仍然合适
        # 反之，将fourier aug去掉，将增强后的标签数据，送入网络，可能会得到好的效果。

        imgs = imgs.to(device)
        aug_imgs = aug_imgs.to(device)
        mask_type = torch.long
        mask = mask.to(device=device, dtype=mask_type)

        unsup_imgs_0 = unsup_imgs_0.to(device)
        unsup_imgs_1 = unsup_imgs_1.to(device)
        aug_unsup_imgs_0 = aug_unsup_imgs_0.to(device)
        aug_unsup_imgs_1 = aug_unsup_imgs_1.to(device)

        mask_params = mask_params.to(device)

        batch_mix_masks = mask_params

        # unlabeled mixed images
        # point4-----------
        # 这里可以针对mixmatch的文章，进行二次修改，mixmatch同时对label和unlabel data做数据增强
        # 结果要好于仅仅对unlabeled data做数据增强，这里可以在分割中尝试
        # 那这里就解释通了，对于unlabeled　data,采用了mixed的思路，同时做了数据增强

        l = np.random.beta(config['alpha'], config['alpha'])
        l = max(l, 1-l)
        batch_mix_masks = l

        unsup_imgs_mixed = unsup_imgs_0 * batch_mix_masks + unsup_imgs_1 * (1 - batch_mix_masks)
        # unlabeled r mixed images
        aug_unsup_imgs_mixed = aug_unsup_imgs_0 * batch_mix_masks + aug_unsup_imgs_1 * (1 - batch_mix_masks)

        # add uncertainty
        # this step is to generate pseudo labels

        with torch.no_grad():
            # Estimate the pseudo-label with model_l using original data
            # 这里得到伪标签没有用到梯度，不必更新

            logits_u0, _ = model(unsup_imgs_0)
            logits_u1, _ = model(unsup_imgs_1)

            logits_u0 = logits_u0.detach()
            logits_u1 = logits_u1.detach()

            aug_logits_u0, _ = model(aug_unsup_imgs_0)
            aug_logits_u1, _ = model(aug_unsup_imgs_1)

            aug_logits_u0 = aug_logits_u0.detach()
            aug_logits_u1 = aug_logits_u1.detach()

        # the augmented data is used to calculate the average pseudo label
        logits_u0 = torch.softmax(logits_u0, dim=1) + torch.softmax(aug_logits_u0, dim=1) / 2
        logits_u1 = torch.softmax(logits_u1, dim=1) + torch.softmax(aug_logits_u1, dim=1) / 2

        pt_u0 = logits_u0 ** (1 / config['T'])
        logits_u0 = pt_u0 / pt_u0.sum(dim=1, keepdim=True)
        logits_u0 = logits_u0.detach()

        pt_u1 = logits_u1 ** (1 / config['T'])
        logits_u1 = pt_u1 / pt_u1.sum(dim=1, keepdim=True)
        logits_u1 = logits_u1.detach()

        # Mix teacher predictions using same mask
        # It makes no difference whether we do this with logits or probabilities as
        # the mask pixels are either 1 or 0

        l = np.random.beta(config['alpha'], config['alpha'])
        l = max(l, 1-l)
        batch_mix_masks = l

        logits_cons = logits_u0 * batch_mix_masks + logits_u1 * (1 - batch_mix_masks)
        _, ps_label = torch.max(logits_cons, dim=1)
        ps_label = ps_label.long()

        # --------------------------------------------------

        num_labeled = len(imgs)

        image_all = torch.cat((imgs, unsup_imgs_mixed))
        pred_all, rep_all = model(image_all)
        # print(rep_all.size())
        pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]

        # ----------supervised loss on both models----------

        sof = F.softmax(pred_l, dim=1)
        loss_sup = total_dice_loss(sof, mask)

        # ---------unlabeled loss---------

        with torch.no_grad():

            pred_all_t, rep_all_t = model(image_all)
            prob_all_t = F.softmax(pred_all_t, dim=1)
            prob_l_t, prob_u_t = (prob_all_t[:num_labeled], prob_all_t[num_labeled:])
            pred_u_t = pred_all_t[num_labeled:]

        # ----------supervised loss for unlabeled sample with low-entropy pseudo label----------

        drop_percent = default_config['drop_percent']
        percent_unreliable = (100 - drop_percent) * (1 - epoch / default_config['num_epoch'])
        drop_percent = 100 - percent_unreliable

        unsup_loss = compute_unsupervised_loss(pred_u, ps_label.clone(), drop_percent, pred_u_t.detach())

        # -----------contrastive loss using unreliable pixels----------

        low_rank, high_rank = default_config['low_rank'], default_config['high_rank']
        alpha_t = default_config['low_entropy_threshold'] * (1 - epoch / default_config['num_epoch'])

        with torch.no_grad():

            prob = torch.softmax(pred_u_t, dim=1)
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

            low_thresh = np.percentile(entropy[ps_label != 255].cpu().numpy().flatten(), alpha_t)
            low_entropy_mask = (entropy.le(low_thresh).float() * (ps_label != 255).bool())

            high_thresh = np.percentile(entropy[ps_label != 255].cpu().numpy().flatten(), 100-alpha_t)
            high_entropy_mask = (entropy.ge(high_thresh).float() * (ps_label != 255).bool())

            # print(mask.size())
            # print(low_entropy_mask.size())

            # print((mask != 255).size())
            # print((mask.unsqueeze(1) != 255).size())
            # print((mask != 255).size())
            # print(low_entropy_mask.unsqueeze(1).size())
            # print(low_entropy_mask)

            # low_mask_all = torch.cat(
            #     (
            #         (mask.unsqueeze(1) != 255).float(),
            #         low_entropy_mask.unsqueeze(1),
            #     )
            # )
            #
            # high_mask_all = torch.cat(
            #     (
            #         (mask.unsqueeze(1) != 255).float(),
            #         high_entropy_mask.unsqueeze(1),
            #     )
            # )

            class_indice = torch.max(mask, 1)[0]
            # print(class_indice.size())

            low_mask_all = torch.cat(
                (
                    (class_indice.unsqueeze(1) != 255).float(),
                    low_entropy_mask.unsqueeze(1),
                )
            )

            high_mask_all = torch.cat(
                (
                    (class_indice.unsqueeze(1) != 255).float(),
                    high_entropy_mask.unsqueeze(1),
                )
            )

            # label_l = label_onehot(mask, default_config['num_class'])
            label_l = label_onehot(class_indice, default_config['num_class'])
            label_u = label_onehot(ps_label, default_config['num_class'])

            prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                rep_all,
                label_l.long(),
                label_u.long(),
                prob_l_t.detach(),
                prob_u_t.detach(),
                low_mask_all,
                high_mask_all,
                memobank,
                queue_ptrlis,
                queue_size,
                rep_all_t.detach(),
                prototype,
                i_iter
            )

        loss = loss_sup + unsup_loss + contra_loss

        loss = (loss-config['b']).abs() + config['b']

        # print(logits_cons.size())
        # guess the pseudo labels for each pixel
        # _, ps_label_1 = torch.max(logits_cons, dim=1)
        # ps_label_1 = ps_label_1.long()

        # Get student_l prediction for mixed image
        # logits_cons_model, _ = model(unsup_imgs_mixed)
        # aug_logits_cons_model, _ = model(aug_unsup_imgs_mixed)

        # add uncertainty
        # var, exp_var = cal_variance(logits_cons_model, aug_logits_cons_model)
        # print(var.size())
        # print(exp_var.size())

        # cps loss
        # cps_loss = torch.mean(exp_var * cross_criterion(logits_cons_model, ps_label)) + torch.mean(var)
        # probs_u = torch.softmax(logits_cons_model, dim=1)
        # print(probs_u.size())
        # exp_var = exp_var.unsqueeze(1)
        # var = var.unsqueeze(1)

        # cps_loss = torch.mean(exp_var * ((probs_u - logits_cons) ** 2) + torch.mean(var))
        # cps_loss = torch.mean((probs_u - logits_cons) ** 2)
        # --------------------point5---------------------

        # 这里也是接上边一点，上下两个模型分别计算两个对应的伪标签，这个过程是完全分离的
        # 所以这里需要单独对两个模型进行更新优化，实则没有必要。
        # 这里可以比较，是一次性生成两组unlabeled　data,对其进行插值，计算比较好
        # 还是依照mixmatch,只生成一组unlabeled data,然后对标签数据和无标签数据同步进行mixup比较好。

        # cps weight
        # cps_loss = cps_loss * config['CPS_weight'] * linear_rampup(epoch)
        # cps_loss = cps_loss * config['CPS_weight']

        # --------------------------point 6-------------------------

        # 这里是对比损失的加入，依照U2PL,将对比学习对应的损失项构造进来，加入
        # contrastive loss /supervised / unsupervised
        # contrastive loss SupConLoss
        # features means different views
        # feature_l = feature_l.unsqueeze(1)
        # feature_r = feature_r.unsqueeze(1)
        # features = torch.cat((feature_l, feature_r),dim=1)
        # supconloss = SupConLoss()
        # con_loss = supconloss(features)
        # 这里作者并没有将对比损失的loss,加入进去,可以在这里尝试重新构造.
        # con_loss = 1

        # 这里因为使用了两个网络，所以将两个loss都加进去。
        # 这里可以认为矢量图的相加，因为每个loss项都保存了其对应的计算图。

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        # step_schedule.step()
        # default_config['learning_rate'] = optimizer.param_groups[-1]['lr']

        # if epoch == 7:
        #     default_config['learning_rate'] = optimizer.param_groups[-1]['lr'] / 2

        # step_size = 550
        # cycle = np.floor(1 + idx / (2 * step_size))
        # x = np.abs(idx / step_size - 2 * cycle + 1)
        # base_lr = 0.0001
        # max_lr = 0.0001350 - 0.000350 * epoch / 900
        # scale_fn = 1 / pow(2, (cycle - 1))
        # default_config['learning_rate'] = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * scale_fn

        # 这里是两个模型同时加载,更新,比较费资源;是否可以采用mean-teacher, 进行修改

        total_loss.append(loss.item())
        total_loss_sup.append(loss_sup.item())
        total_unsup_loss.append(unsup_loss.item())
        total_con_loss.append(contra_loss.item())

    total_loss = sum(total_loss) / len(total_loss)
    total_loss_sup = sum(total_loss_sup) / len(total_loss_sup)
    total_unsup_loss = sum(total_unsup_loss) / len(total_unsup_loss)
    total_con_loss = sum(total_con_loss) / len(total_con_loss)

    # return model, total_loss, total_loss_sup, total_unsup_loss, total_con_loss, default_config['learning_rate']
    return model, total_loss, total_loss_sup, total_unsup_loss, total_con_loss

# use the function to calculate the valid loss or test loss


def test_dual_dy(model, loader):

    # ----------------------point7--------------------

    # 同理，这里只加载一个model就可以啦，同时加载两个进行training是对资源的浪费
    # 同时，也都利用上了作者所提的两个点。
    model.eval()

    loss = []
    tot = 0
    tot_lv = 0
    tot_myo = 0
    tot_rv = 0

    for batch in tqdm(loader):

        imgs = batch['img']
        mask = batch['mask']
        imgs = imgs.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits, _ = model(imgs)

        pred = F.softmax(logits, dim=1)

        # 这里因为只输出一个结果，所以选择对两个网络输出结果进行平均。

        pred = (pred > 0.5).float()

        # loss
        # 这里应该是对的，其对应的顺序为3个前景，４是背景。
        dice_loss_lv = dice_loss(pred[:, 0, :, :], mask[:, 0, :, :])
        dice_loss_myo = dice_loss(pred[:, 1, :, :], mask[:, 1, :, :])
        dice_loss_rv = dice_loss(pred[:, 2, :, :], mask[:, 2, :, :])
        dice_loss_bg = dice_loss(pred[:, 3, :, :], mask[:, 3, :, :])

        t_loss = dice_loss_lv + dice_loss_myo + dice_loss_rv + dice_loss_bg

        loss.append(t_loss.item())

        # dice score
        tot += dice_coeff(pred[:, 0:3, :, :],
                          mask[:, 0:3, :, :], device).item()
        tot_lv += dice_coeff(pred[:, 0, :, :], mask[:, 0, :, :], device).item()
        tot_myo += dice_coeff(pred[:, 1, :, :],
                              mask[:, 1, :, :], device).item()
        tot_rv += dice_coeff(pred[:, 2, :, :], mask[:, 2, :, :], device).item()

    r_loss = sum(loss) / len(loss)
    dice_lv = tot_lv/len(loader)
    dice_myo = tot_myo/len(loader)
    dice_rv = tot_rv/len(loader)
    dice = tot/len(loader)

    return r_loss, dice, dice_lv, dice_myo, dice_rv


def train_dy(label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, learning_rate, weight_decay,
             ema_decay, num_epoch, model_path, niters_per_epoch):

    global prototype

    # Initialize model
    model, ema_model = ini_model_dy()

    # Initialize optimizer.
    optimizer, ema_optimizer, step_schedule = ini_optimizer_dy(
        model, ema_model, learning_rate, weight_decay, ema_decay)

    # loss
    cross_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    # square_loss = SquareLoss()

    best_dice = 0

    # build class_wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(default_config['num_class']):
        memobank.append([torch.zeros(0,256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[3] = 50000

    prototype = torch.zeros(
        default_config['num_class'],
        default_config['num_queries'],
        1,
        # -----------dim of feature----------
        256,
    ).cuda()

    for epoch in range(num_epoch):

        # ---------- Training ----------

        model.train()

        label_dataloader = iter(label_loader)
        unlabel_dataloader_0 = iter(unlabel_loader_0)
        unlabel_dataloader_1 = iter(unlabel_loader_1)

        # normal images
        model, total_loss, total_loss_sup, total_cps_loss, total_con_loss,  = train_one_epoch_dy(model, niters_per_epoch,
            label_dataloader, unlabel_dataloader_0, unlabel_dataloader_1, optimizer, ema_optimizer, step_schedule,
            cross_criterion, epoch, memobank, queue_ptrlis, queue_size)

        # Print the information.
        print(
            f"[ Normal image Train | {epoch + 1:03d}/{num_epoch:03d} ] learning_rate = {default_config['learning_rate']:.5f}  total_loss = {total_loss:.5f}  total_loss_sup = {total_loss_sup:.5f}  total_cps_loss = {total_cps_loss:.5f}")

        # ---------- Validation----------
        val_loss, val_dice, val_dice_lv, val_dice_myo, val_dice_rv = test_dual_dy(
            ema_model, val_loader)
        print(
            f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] val_loss = {val_loss:.5f} val_dice = {val_dice:.5f}")

        # ---------- Testing (using ensemble)----------
        test_loss, test_dice, test_dice_lv, test_dice_myo, test_dice_rv = test_dual_dy(
            ema_model, test_loader)
        print(
            f"[ Test | {epoch + 1:03d}/{num_epoch:03d} ] test_loss = {test_loss:.5f} test_dice = {test_dice:.5f}")

        # val
        wandb.log({'val/val_dice': val_dice, 'val/val_dice_lv': val_dice_lv,
                  'val/val_dice_myo': val_dice_myo, 'val/val_dice_rv': val_dice_rv})
        # test
        wandb.log({'test/test_dice': test_dice, 'test/test_dice_lv': test_dice_lv,
                  'test/test_dice_myo': test_dice_myo, 'test/test_dice_rv': test_dice_rv})
        # loss
        wandb.log({'epoch': epoch + 1, 'learning_rate': default_config['learning_rate'],
                   'loss/total_loss': total_loss, 'loss/total_loss_sup': total_loss_sup,
                   'loss/total_cps_loss': total_cps_loss, 'loss/test_loss': test_loss,
                   'loss/val_loss': val_loss, 'loss/con_loss': total_con_loss})

        # if the model improves, save a checkpoint at this epoch
        if val_dice > best_dice:
            best_dice = val_dice
            print('saving model with best_dice {:.5f}'.format(best_dice))
            model_name = './tmodel/u2pl/' + model_path
            torch.save(model.module, model_name)


def main():

    batch_size = config['batch_size']
    num_workers = config['num_workers']
    learning_rate = default_config['learning_rate']
    weight_decay = config['weight_decay']
    ema_decay = config['ema_decay']
    num_epoch = config['num_epoch']
    model_path = config['model_path']
    test_vendor = config['test_vendor']

    label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, num_label_imgs, num_unsup_imgs = \
        pre_data(batch_size=batch_size, num_workers=num_workers, test_vendor=test_vendor)

    max_samples = num_unsup_imgs
    niters_per_epoch = int(math.ceil(max_samples * 1.0 // batch_size))

    print("max_samples", max_samples)
    print("niters_per_epoch", niters_per_epoch)

    if config['Fourier_aug']:
        print("Fourier mode")
    #     Fourier mode for data augmentation
    else:
        print("Normal mode")

    train_dy(label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, learning_rate,
             weight_decay, ema_decay, num_epoch, model_path, niters_per_epoch)


if __name__ == '__main__':
    main()
