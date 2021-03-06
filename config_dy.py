# name = 'MMS_deeplab_5%_A_CM_DY'
name = 'MMS_deeplab_5%_B_CM_DY'
# name = 'MMS_deeplab_5%_C_CM_DY'
# name = 'MMS_deeplab_5%_D_CM_DY'


# hyperparameter
default_config = dict(
    batch_size=8,
    num_epoch=50,
    learning_rate=1e-4,            # learning rate of Adam
    weight_decay=0.01,             # weight decay 
    num_workers=8,
    ema_decay=0.999,
    T=0.5,
    alpha=0.75,
    # stop training to avoid overfitting
    b=0.23150,
    num_class=4,
    # unsupervised
    # 取熵最大的前20%作为unreliable unlabeled data.
    # 取剩余的80%作为reliable labeled data.
    drop_percent=80,

    # contrastive loss
    num_negatives=50,
    num_queries=256,
    temperature=0.5,

    train_name=name,
    model_path=name+'.pt',
    test_vendor='B',
    ratio=0.05,                   # 2%
    # this parameter can be revised based on k-fold validation
    CPS_weight=3,

    gpus=[4],
    ifFast=False,
    Pretrain=True,
    # pretrain_file='/home/duyao/my_data/duyao/MMData/resnet50_v1c.pth',
    pretrain_file='/home/listu/yaodu/MMData/resnet50_v1c.pth',

    restore=False,
    restore_from=name+'.pt',

    # for cutmix
    cutmix_mask_prop_range=(0.25, 0.5),
    cutmix_boxmask_n_boxes=3,
    cutmix_boxmask_fixed_aspect_ratio=True,
    cutmix_boxmask_by_size=True,
    cutmix_boxmask_outside_bounds=True,
    cutmix_boxmask_no_invert=True,

    Fourier_aug=True,
    fourier_mode='AS',

    # for bn
    bn_eps=1e-5,
    bn_momentum=0.1,
)
