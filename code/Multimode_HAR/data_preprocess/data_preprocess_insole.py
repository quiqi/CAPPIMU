import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle as cp
from data_preprocess.base_loader import base_loader, data_root
from data_preprocess.data_preprocess_utils import *
from utils.util import args_contains, seed_worker


class data_loader_insole(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_insole, self).__init__(samples, labels, domains)

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain


def apply_label_map(y):
    """
    将字符标签转换成数字标签
    {'fall': 0,'run': 1,'walk': 2,'cycle': 3,'lay': 4,'squat': 5,'mop': 6,'drink': 7,'sweep': 8,'brushing_teeth': 9,
    'cut': 10,'eat': 11,'folding_clothes': 12,'hang_out_clothes': 13,'ironing': 14,'open_door': 15,'open_fridge': 16,
    'sit': 17,'stand': 18,'use_computer': 19,'wash_dish': 20,'wash_face': 21,'wash_window': 22,'watch_tv': 23,
    'watering_flowers': 24,'write': 25,'wc': 26,'play_phone': 27,'switch': 28}
    Parameters:
        y: 1D array of labels

    Return:
        y_mapped: 1D array of mapped labels
    """
    label_list = ['fall', 'run', 'walk', 'cycle', 'brushing_teeth', 'cut', 'eat', 'folding_clothes', 'use_computer',
                  'wash_dish', 'wash_face', 'write', 'play_phone', 'sweep', 'mop', 'wc', 'wash_window', 'drink',
                  'watch_tv', 'hang_out_clothes', 'ironing']
    label_map = dict([(l, i) for i, l in enumerate(label_list)])

    y_mapped = []
    for label in y:
        y_mapped.append(label_map.get(label))
    return np.array(y_mapped)


def process_dataset_file(args, data):
    # 将data的数据和标签分离出来
    start_x = 0
    end_x = args.n_feature
    data_y = data[:, end_x]
    data_x = data[:, start_x:end_x]
    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    # 将字符标签转换成数字标签
    data_y = apply_label_map(data_y)
    data_y = data_y.astype(int)

    return data_x, data_y


def load_data_files(args, data_files):
    user_trial_dataset = pd.read_csv(data_files, dtype={"time": object})
    # 删除没有标签的行
    user_trial_dataset.dropna(axis=0, subset=['label'], inplace=True)
    # 空值判断
    if np.all(user_trial_dataset.isnull()):
        print('[ERROR] This file has null value!')
    # 1*time + 14_l + 14_r + label
    end_x = 1 + args.n_feature + 1
    data = user_trial_dataset[user_trial_dataset.columns[1:end_x]].to_numpy()
    print('... file {0}'.format(data_files))
    x, y = process_dataset_file(args, data)

    return x, y


def process_row_value_files(args, domain_idx):
    """
    按照参数user_id读入原始文件，保存成.data文件
    :param args:
    :param domain_idx: user_id
    :return: values, labels, user_id
    """
    data_dir = data_root + '/insole_new_label/'

    saved_filename = data_dir + args.device + '_domain_' + domain_idx + '_wd.data'  # "wd": with domain label

    if os.path.isfile(saved_filename):
        data = np.load(saved_filename, allow_pickle=True)
        values = data[0][0]
        labels = data[0][1]
        user_id = data[0][2]
        return values, labels, user_id
    else:
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        cur_domain_files = data_dir + 'subject_' + domain_idx
        if len(args.device) > 8:
            cur_domain_files = data_dir + 'subject_' + domain_idx + '/joint/'
        for trial_user_file in sorted(glob.glob(cur_domain_files + '/*.csv')):
            device_name = os.path.split(trial_user_file)[-1].split(".")[0]
            if device_name == args.device:
                values, labels = load_data_files(args, trial_user_file)
                user_id = np.full(labels.shape, (int(domain_idx)) % 100, dtype=int)
                obj = [(values, labels, user_id)]
                f = open(os.path.join(saved_filename), 'wb')
                cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
                f.close()

        return values, labels, user_id


# 不跨人数据划分
def pre_process_dataset_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                          '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
    # 服务器上运行可以更改
    workers = args_contains(args, 'workers', 0)
    pin_memory = args_contains(args, 'pin_memory', True)

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for index, source_domain in enumerate(source_domain_list):
        single_x, single_y, domain = process_row_value_files(args, source_domain)
        x_win, y_win, d_win = opp_sliding_window_w_d(single_x, single_y, domain, SLIDING_WINDOW_LEN,
                                                     SLIDING_WINDOW_STEP)
        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win

    print("x_win_all: {}, y_win_all: {}, d_win_all: {}".format(x_win_all.shape, y_win_all.shape, d_win_all.shape))
    x_win_train, x_win_test, y_win_train, y_win_test, d_win_train, d_win_test = \
        train_test_val_split(x_win_all, y_win_all, d_win_all, split_ratio=args.split_ratio)

    # 首先对训练数据去重，然后求训练数据的均值和标准差
    s = cut_overlap(x_win_train)
    x_win_train = normalize(s, x_win_train)
    x_win_test = normalize(s, x_win_test)

    print("x_win_train: {}, x_win_test: {}, y_win_train: {}, y_win_test: {}, "
          "d_win_train: {}, d_win_test: {}"
          .format(x_win_train.shape, x_win_test.shape, y_win_train.shape, 
                  y_win_test.shape, d_win_train.shape, d_win_test.shape))

    # 保存划分好的窗口数据
    # np.save('/home/intelligence/Robin/Dataset/save_raw_data/' + args.device + '_train_win_x.npy', x_win_train)
    # np.save('/home/intelligence/Robin/Dataset/save_raw_data/' + args.device + '_train_win_y.npy', y_win_train)
    # np.save('/home/intelligence/Robin/Dataset/save_raw_data/' + args.device + '_test_win_x.npy', x_win_test)
    # np.save('/home/intelligence/Robin/Dataset/save_raw_data/' + args.device + '_test_win_y.npy', y_win_test)
    # np.save('/home/intelligence/Robin/Dataset/save_raw_data/' + args.device + '_train_win_d.npy', d_win_train)
    # np.save('/home/intelligence/Robin/Dataset/save_raw_data/' + args.device + '_test_win_d.npy', d_win_test)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))

    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights),
                                                             replacement=True)
    # train_set_r[0][0]
    train_set_r = data_loader_insole(x_win_train, y_win_train, d_win_train)
    generator = torch.Generator()
    generator.manual_seed(0)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                                num_workers=workers, worker_init_fn=seed_worker, generator=generator, pin_memory=pin_memory)

    test_set_r = data_loader_insole(x_win_test, y_win_test, d_win_test)
    generator_test = torch.Generator()
    generator_test.manual_seed(2023)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                               worker_init_fn=seed_worker, generator=generator_test, pin_memory=pin_memory)

    return [train_loader_r], test_loader_r


# 跨人数据划分
def pre_process_dataset_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP, train_user, test_user):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                          '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
    workers = args_contains(args, 'workers', 0)
    pin_memory = args_contains(args, 'pin_memory', False)

    # train_single_x_all, train_single_y_all, test_single_x_all, test_single_y_all = np.array([]), np.array([]), np.array([]), np.array([])
    train_x_win_all, train_y_win_all, train_d_win_all = np.array([]), np.array([]), np.array([])
    test_x_win_all, test_y_win_all, test_d_win_all = np.array([]), np.array([]), np.array([])
    train_user_id, test_user_id = [], []
    for user in source_domain_list:
        if user in train_user:
            train_single_x, train_single_y, train_domain = process_row_value_files(args, user)
            # print("train_single_x: {}, train_single_y: {}".format(train_single_x.shape, train_single_y.shape))

            # train_single_x_all = np.concatenate((train_single_x_all, train_single_x), axis=0) if train_single_x_all.size else train_single_x
            # train_single_y_all = np.concatenate((train_single_y_all, train_single_y), axis=0) if train_single_y_all.size else train_single_y

            train_x_win, train_y_win, train_d_win = opp_sliding_window_w_d(train_single_x, train_single_y, train_domain,
                                                                           SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
            train_x_win_all = np.concatenate((train_x_win_all, train_x_win), axis=0) if train_x_win_all.size else train_x_win
            train_y_win_all = np.concatenate((train_y_win_all, train_y_win), axis=0) if train_y_win_all.size else train_y_win
            train_d_win_all = np.concatenate((train_d_win_all, train_d_win), axis=0) if train_d_win_all.size else train_d_win
            train_user_id.append(user)
        elif user in test_user:
            test_single_x, test_single_y, test_domain = process_row_value_files(args, user)
            # print("test_single_x: {}, test_single_y: {}".format(test_single_x.shape, test_single_y.shape))

            # test_single_x_all = np.concatenate((test_single_x_all, test_single_x),
            #                                     axis=0) if test_single_x_all.size else test_single_x
            # test_single_y_all = np.concatenate((test_single_y_all, test_single_y),
            #                                     axis=0) if test_single_y_all.size else test_single_y

            test_x_win, test_y_win, test_d_win = opp_sliding_window_w_d(test_single_x, test_single_y, test_domain,
                                                                        SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
            test_x_win_all = np.concatenate((test_x_win_all, test_x_win), axis=0) if test_x_win_all.size else test_x_win
            test_y_win_all = np.concatenate((test_y_win_all, test_y_win), axis=0) if test_y_win_all.size else test_y_win
            test_d_win_all = np.concatenate((test_d_win_all, test_d_win), axis=0) if test_d_win_all.size else test_d_win
            test_user_id.append(user)
        else:
            print('Please input correct user_ID')

    s = cut_overlap(train_x_win_all)
    train_x_win_all = normalize(s, train_x_win_all)
    test_x_win_all = normalize(s, test_x_win_all)

    print("train_user:{}".format(train_user_id))
    print("test_user:{}".format(test_user_id))
    print("x_win_train: {}, x_win_test: {}, y_win_train: {}, y_win_test: {}, ""d_win_train: {}, d_win_test: {}"
          .format(train_x_win_all.shape, test_x_win_all.shape, train_y_win_all.shape,
                  test_y_win_all.shape, train_d_win_all.shape, test_d_win_all.shape))

    # 保存划分好的窗口数据
    # np.save('/home/intelligence/Robin/Dataset/train_all_x.npy', train_single_x_all)
    # np.save('/home/intelligence/Robin/Dataset/train_all_y.npy', train_single_y_all)
    # np.save('/home/intelligence/Robin/Dataset/test_all_x.npy', test_single_x_all)
    # np.save('/home/intelligence/Robin/Dataset/test_all_y.npy', test_single_y_all)

    unique_y, counts_y = np.unique(train_y_win_all, return_counts=True)
    unique_test_y, counts_test_y = np.unique(test_y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    print('y_test label distribution: ', dict(zip(unique_test_y, counts_test_y)))

    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(train_y_win_all, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights),
                                                             replacement=True)
    # 当sampler有输入时，shuffle的值就没有意义
    train_set_r = data_loader_insole(train_x_win_all, train_y_win_all, train_d_win_all)
    generator = torch.Generator()
    generator.manual_seed(0)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                                num_workers=workers, worker_init_fn=seed_worker, generator=generator, pin_memory=pin_memory)
    test_set_r = data_loader_insole(test_x_win_all, test_y_win_all, test_d_win_all)
    generator_test = torch.Generator()
    generator_test.manual_seed(2023)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                               worker_init_fn=seed_worker, generator=generator_test, pin_memory=pin_memory)

    return [train_loader_r], test_loader_r


def process_row_device_files(args, device, domain_idx):
    """
    按照参数user_id读入原始文件，保存成.data文件
    :param args:
    :param device:
    :param domain_idx: user_id
    :return: values, labels, user_id
    """
    data_dir = data_root + '/insole_new_label/'

    print('\nProcessing domain {0} files...\n'.format(domain_idx))
    # cur_domain_files = data_dir + 'subject_' + domain_idx + '/joint/'
    cur_domain_files = data_dir + 'subject_' + domain_idx
    for trial_user_file in sorted(glob.glob(cur_domain_files + '/*.csv')):
        device_name = os.path.split(trial_user_file)[-1].split(".")[0]
        if device_name == device:
            values, labels = load_data_files(args, trial_user_file)
            user_id = np.full(labels.shape, (int(domain_idx)) % 100, dtype=int)

    return values, labels, user_id


# 跨位置数据划分
def pre_process_dataset_random_device(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP):
    source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                          '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
    # source_domain_list = ['2']
    # 服务器上运行可以更改
    workers = args_contains(args, 'workers', 0)
    pin_memory = args_contains(args, 'pin_memory', True)

    # 提取训练集传感器位置数据
    train_x_win_all, train_y_win_all, train_d_win_all = np.array([]), np.array([]), np.array([])
    for index, source_domain in enumerate(source_domain_list):
        train_single_x, train_single_y, train_domain = process_row_device_files(args, args.train_device, source_domain)

        train_x_win, train_y_win, train_d_win = opp_sliding_window_w_d(train_single_x, train_single_y, train_domain,
                                                                       SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        train_x_win_all = np.concatenate((train_x_win_all, train_x_win), axis=0) if train_x_win_all.size else train_x_win
        train_y_win_all = np.concatenate((train_y_win_all, train_y_win), axis=0) if train_y_win_all.size else train_y_win
        train_d_win_all = np.concatenate((train_d_win_all, train_d_win), axis=0) if train_d_win_all.size else train_d_win

    x_win_train, _, y_win_train, _, d_win_train, _ = train_test_val_split(train_x_win_all, train_y_win_all,
                                                                          train_d_win_all, split_ratio=args.split_ratio)

    print("x_win_train: {}, y_win_train: {}, d_win_train: {}\n"
          .format(x_win_train.shape, y_win_train.shape, d_win_train.shape))

    # 提取测试集传感器位置数据
    test_x_win_all, test_y_win_all, test_d_win_all = np.array([]), np.array([]), np.array([])
    for index, source_domain in enumerate(source_domain_list):
        test_single_x, test_single_y, test_domain = process_row_device_files(args, args.test_device, source_domain)

        test_x_win, test_y_win, test_d_win = opp_sliding_window_w_d(test_single_x, test_single_y, test_domain,
                                                                       SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        test_x_win_all = np.concatenate((test_x_win_all, test_x_win), axis=0) if test_x_win_all.size else test_x_win
        test_y_win_all = np.concatenate((test_y_win_all, test_y_win), axis=0) if test_y_win_all.size else test_y_win
        test_d_win_all = np.concatenate((test_d_win_all, test_d_win), axis=0) if test_d_win_all.size else test_d_win

    _, x_win_test, _, y_win_test, _, d_win_test = \
        train_test_val_split(test_x_win_all, test_y_win_all, test_d_win_all, split_ratio=args.split_ratio)

    print("x_win_test: {}, y_win_test: {}, d_win_test: {}\n"
          .format(x_win_test.shape, y_win_test.shape, d_win_test.shape))

    print(f"source_domain:{source_domain} | train_device:{args.train_device} | train_device:{args.test_device}\n")

    # 首先对训练数据去重，然后求训练数据的均值和标准差
    s = cut_overlap(x_win_train)
    x_win_train = normalize(s, x_win_train)
    x_win_test = normalize(s, x_win_test)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    unique_test_y, counts_test_y = np.unique(y_win_test, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    print('y_test label distribution: ', dict(zip(unique_test_y, counts_test_y)))

    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights),
                                                             replacement=True)
    # 当sampler有输入时，shuffle的值就没有意义
    train_set_r = data_loader_insole(x_win_train, y_win_train, d_win_train)
    generator = torch.Generator()
    generator.manual_seed(0)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler,
                                num_workers=workers, worker_init_fn=seed_worker, generator=generator, pin_memory=pin_memory)

    test_set_r = data_loader_insole(x_win_test, y_win_test, d_win_test)
    generator_test = torch.Generator()
    generator_test.manual_seed(2023)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                               worker_init_fn=seed_worker, generator=generator_test, pin_memory=pin_memory)

    return [train_loader_r], test_loader_r


def prep_insole(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0, train_user=None, test_user=None):
    if args.cases == 'random':
        return pre_process_dataset_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject':
        return pre_process_dataset_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP, train_user, test_user)
    elif args.cases == 'random_device':
        return pre_process_dataset_random_device(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    else:
        return 'Error! Unknown args.cases!\n'

