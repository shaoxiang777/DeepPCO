from pathlib import Path

args = {
    # file path for pc
    # 'root': '.',
    'euroc_path': '/home/ies/zhu/project/datasets',
    'kitti_path': '/home/ies/zhu/project/datasets',
    'save_path': '/home/ies/zhu/project/models',

    # net parameters
    'flownet_path': "/home/ies/zhu/project/models/flownetc_EPE1.766.tar",
    'pretrained': True,
    'dropout_rate': 0.4,
    'fc_ks_euroc': [(11, 24), (4, 7)],
    'fc_ks_kitti': [(3, 63), (2, 17)],

    # dataset parameters
    'input_size': (400, 200),
    'train_split': ['MH_01_easy'
        ,'MH_03_medium',
                    'MH_04_difficult', 'MH_05_difficult',
                    'V1_02_medium',
                    'V1_03_difficult',
                    'V2_01_easy',
                    'V2_02_medium'],
    'test_split':  [
        'V1_01_easy',
        'MH_02_easy'],

    # 'train_split': ['00', '02', '03', '05',
    #                 '06', '08', '09', '04', '01'],
    # # 'test_split':  ['07', '10'],
    # 'test_split':  ['10'],

    'train_batch_size': 32,
    'num_workers': 8,
    'shuffle': True,
    'reverse': True,
    'hflip': True,

    # train parameters
    "loss_weight": 100,
    'optim': 'Adam',  # SGD
    'epoch': 55,
    'lr': 0.0001,
    'weight_decay': 0.5,
    'momentum': 0.9,

    # load net.pth
    'net_path': "/home/ies/zhu/project/models/2021-02-11-02-04-44/net_parameters_epoch_53.pth",
}
