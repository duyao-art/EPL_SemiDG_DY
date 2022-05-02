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
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models

from network.network import my_net
from utils.utils import get_device, check_accuracy, check_accuracy_dual, label_to_onehot
from mms_dataloader_dy import get_meta_split_data_loaders
from config_dy import default_config
from utils.dice_loss import dice_coeff
# from losses import SupConLoss
import utils.mask_gen as mask_gen
from utils.custom_collate import SegCollate

# multiple GPU setting
gpus = default_config['gpus']
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

wandb.init(project='MNMS_SemiDG_DY', entity='du-yao',
           config=default_config, name=default_config['train_name'])
config = wandb.config


device = get_device()

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

    model = model.to(device)
    model.device = device

    ema_model = ema_model.to(device)
    ema_model.device = device

    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    ema_model = nn.DataParallel(ema_model, device_ids=gpus, output_device=gpus[0])

    return model, ema_model


def create_model(ema=False, restore=False, restore_from=None):

    if restore:
        model_path = './tmodel/' + str(restore_from)
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

    # if epoch == 5:
    #     learning_rate = learning_rate / 2
    # Initialize two optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    step_schedule = torch.optim.lr_scheduler.StepLR(step_size=5, gamma=0.9, optimizer=optimizer)
    # step_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=15)
    # 问题出在哪里呢？

    ema_optimizer = WeightEMA(model, ema_model, alpha=ema_decay)

    return optimizer, ema_optimizer, step_schedule


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


def train_one_epoch_dy(model, niters_per_epoch, label_dataloader, unlabel_dataloader_0, unlabel_dataloader_1, optimizer,
                       ema_optimizer, step_schedule, cross_criterion, epoch):

    # loss data
    total_loss = []
    total_loss_sup = []
    total_cps_loss = []
    total_con_loss = []

    # tqdm
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(niters_per_epoch),
                file=sys.stdout, bar_format=bar_format)

    # 每个batch进行更新
    for idx in pbar:

        minibatch = label_dataloader.next()
        unsup_minibatch_0 = unlabel_dataloader_0.next()
        unsup_minibatch_1 = unlabel_dataloader_1.next()

        # Multiple dictionary in self-defined Dataset
        # 自定义的dataset,返回的值是一个字典，从原始数据到增强的数据。

        # --------------------point7 --------------------

        # 这里的fourier augmentation已经是一个非常强的变换了
        # 可以考虑在第二次实验中，是否对label，unlabel统一做一次mixup,而不是只针对unlabeled data.

        imgs = minibatch['img']
        aug_imgs = minibatch['aug_img']
        mask = minibatch['mask']

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

        # print(logits_cons.size())
        # guess the pseudo labels for each pixel
        # _, ps_label_1 = torch.max(logits_cons, dim=1)
        # ps_label_1 = ps_label_1.long()

        # Get student_l prediction for mixed image
        logits_cons_model, _ = model(unsup_imgs_mixed)
        aug_logits_cons_model, _ = model(aug_unsup_imgs_mixed)

        # add uncertainty
        var, exp_var = cal_variance(logits_cons_model, aug_logits_cons_model)
        # print(var.size())
        # print(exp_var.size())

        # cps loss
        cps_loss = torch.mean(exp_var * cross_criterion(logits_cons_model, ps_label)) + torch.mean(var)
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

        # -------这个loss的权重是不是太小了，导致无标签数据对应的损失在更新中所占的比例太小。　
        cps_loss = cps_loss * config['CPS_weight']

        # supervised loss on both models
        # 这里对原始有标签image，做监督学习对应的损失
        pre_sup, feature = model(imgs)

        # dice loss
        sof = F.softmax(pre_sup, dim=1)
        loss_sup = total_dice_loss(sof, mask)

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
        con_loss = 1

        optimizer.zero_grad()

        # 这里因为使用了两个网络，所以将两个loss都加进去。
        # 这里可以认为矢量图的相加，因为每个loss项都保存了其对应的计算图。

        loss = loss_sup + cps_loss
        loss = (loss-config['b']).abs() + config['b']

        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        step_schedule.step()
        default_config['learning_rate'] = optimizer.param_groups[-1]['lr']
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
        total_cps_loss.append(cps_loss.item())
        total_con_loss.append(con_loss)

    total_loss = sum(total_loss) / len(total_loss)
    total_loss_sup = sum(total_loss_sup) / len(total_loss_sup)
    total_cps_loss = sum(total_cps_loss) / len(total_cps_loss)
    total_con_loss = sum(total_con_loss) / len(total_con_loss)

    return model, total_loss, total_loss_sup, total_cps_loss, total_con_loss, default_config['learning_rate']


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

    # Initialize model
    model, ema_model = ini_model_dy()

    # Initialize optimizer.
    optimizer, ema_optimizer, step_schedule = ini_optimizer_dy(
        model, ema_model, learning_rate, weight_decay, ema_decay)

    # loss
    cross_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    # square_loss = SquareLoss()

    best_dice = 0

    for epoch in range(num_epoch):

        # ---------- Training ----------
        model.train()

        label_dataloader = iter(label_loader)
        unlabel_dataloader_0 = iter(unlabel_loader_0)
        unlabel_dataloader_1 = iter(unlabel_loader_1)

        # normal images
        model, total_loss, total_loss_sup, total_cps_loss, total_con_loss, default_config['learning_rate'] = \
            train_one_epoch_dy(model, niters_per_epoch, label_dataloader, unlabel_dataloader_0, unlabel_dataloader_1,
                               optimizer, ema_optimizer, step_schedule, cross_criterion, epoch)

        # Print the information.
        print(
            f"[ Normal image Train | {epoch + 1:03d}/{num_epoch:03d} ] learning_rate = {default_config['learning_rate']:.8f}  total_loss = {total_loss:.5f}  total_loss_sup = {total_loss_sup:.5f}  total_cps_loss = {total_cps_loss:.5f}")

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
            model_name = './tmodel/' + model_path
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
