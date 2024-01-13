import argparse


def init_parameters(args=None):
    parser = argparse.ArgumentParser(description='argument setting of network')

    parser.add_argument('--cuda', default=0, type=int, help='cuda device ID，0/1')
    # hyperparameter
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
    parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    # dataset
    parser.add_argument('--dataset', type=str, default='insole', choices=['pamap2', 'ucihar', 'insole'],
                        help='name of dataset')
    parser.add_argument('--n_feature', type=int, default=28, help='name of feature dimension')
    parser.add_argument('--len_sw', type=int, default=200, help='length of sliding window')
    parser.add_argument('--n_class', type=int, default=21, help='number of class')
    parser.add_argument('--cases', type=str, default='random', choices=['random', 'subject', 'random_device'],
                        help='name of scenarios')
    parser.add_argument('--split_ratio', type=float, default=0.3,
                        help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')

    # backbone model
    parser.add_argument('--backbone', type=str, default='FCN',
                        choices=['FCN', 'DCL', 'TPN', 'Transformer', 'resnet', 'DCLS', 'TA'], help='name of framework')

    # log
    parser.add_argument('--logdir', type=str, default='log/', help='log directory')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='the optimizer for model training')

    # Select Device Type
    parser.add_argument('--device', type=str, default='insole', choices=['insole', 'pocket', 'arm_l', 'arm_r', 'wrist_l',
                        'wrist_r', 'chest', 'knee_l', 'knee_r', 'head', 'insole_head', 'insole_arm_l', 'insole_arm_r',
                        'insole_wrist_l', 'insole_wrist_r', 'insole_chest', 'insole_knee_l', 'insole_knee_r',
                                                                         'insole_pocket', 'None'], help='')
    # parser.add_argument('--train_device', type=str, default='None',
    #                     choices=['insole_head', 'insole_arm_l', 'insole_arm_r', 'insole_wrist_l', 'insole_wrist_r',
    #                              'insole_chest', 'insole_knee_l', 'insole_knee_r', 'insole_pocket', 'None'], help='')
    # parser.add_argument('--test_device', type=str, default='None',
    #                     choices=['insole_head', 'insole_arm_l', 'insole_arm_r', 'insole_wrist_l', 'insole_wrist_r',
    #                              'insole_chest', 'insole_knee_l', 'insole_knee_r', 'insole_pocket', 'None'], help='')
    parser.add_argument('--train_device', type=str, default='None',
                        choices=['head', 'arm_l', 'arm_r', 'wrist_l', 'wrist_r',
                                 'chest', 'knee_l', 'knee_r', 'pocket', 'None'], help='')
    parser.add_argument('--test_device', type=str, default='None',
                        choices=['head', 'arm_l', 'arm_r', 'wrist_l', 'wrist_r',
                                 'chest', 'knee_l', 'knee_r', 'pocket', 'None'], help='')

    parser.add_argument('--workers', default=0, type=int)
    # pin_memory
    parser.add_argument('--pin_memory', default=False, type=bool)
    
    # 跨域k折交叉验证
    parser.add_argument('--fold', default=0, type=int, help='range: 0 ~ k_fold-1')
    # 跨人改为30 跨设备改为81
    parser.add_argument('--k_fold', default=81, type=int, help='the number of k in k-fold cross validation')

    # labels-----如果修改了标签记得删除原来生成的data文件
    parser.add_argument('--labels', type=str, nargs='+', default=['falling', 'jogging', 'walking', 'cycling',
                                                                  'brushing_teeth', 'slicing', 'eating',
                                                                  'folding_clothes', 'using_the_computer',
                                                                  'washing_dishes', 'washing_face', 'writing',
                                                                  'play_with_phone', 'sweeping', 'mopping', 'toileting',
                                                                  'window_cleaning', 'drinking_water', 'watching_tv',
                                                                  'hanging_out_clothes', 'ironing'], help='')

    # 专为UCI和PAMAP2用
    parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                       '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,22,23,24,25,29] for shar, '
                                                                       '[a-i] for hhar')
    if args is None:
        args = ""
    return parser.parse_args(args=args)
