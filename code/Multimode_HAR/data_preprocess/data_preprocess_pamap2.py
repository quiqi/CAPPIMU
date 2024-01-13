import numpy as np
import torch
import pickle as cp
from torch.utils.data import Dataset, DataLoader
from data_preprocess.data_preprocess_utils import *
from data_preprocess.base_loader import base_loader, data_root
from utils.util import args_contains

NUM_FEATURES = 52
# 3个位置， 每个位置17个数据
# NUM_FEATURES = 17
NUM_TIMESTEPS = 170
NUM_CLASS = 12


class data_loader_pamap2(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_pamap2, self).__init__(samples, labels, domains)

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain


def complete_HR(data):
    """Sampling rate for the heart rate is different from the other sensors. Missing
    measurements are filled

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        HR channel data
    """

    pos_NaN = np.isnan(data)
    idx_NaN = np.where(pos_NaN == False)[0]
    data_no_NaN = data * 0
    for idx in range(idx_NaN.shape[0] - 1):
        data_no_NaN[idx_NaN[idx]: idx_NaN[idx + 1]] = data[idx_NaN[idx]]

    data_no_NaN[idx_NaN[-1]:] = data[idx_NaN[-1]]

    return data_no_NaN


def divide_x_y(data, position=None):
    """Segments each sample into time, labels and sensor channels

    :param position: 传感器位置 手 胸 脚踝
    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        Time and labels as arrays, sensor channels as matrix
    """
    if position == 'hand':
        start_x = 3
        end_x = 20
    elif position == 'chest':
        start_x = 20
        end_x = 37
    elif position == 'Ankle':
        start_x = 37
        end_x = 54
    else:
        start_x = 2
        end_x = 54
    data_t = data[:, 0]
    data_y = data[:, 1]
    data_x = data[:, start_x:end_x]

    return data_t, data_x, data_y


def adjust_idx_labels(data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function adjust the labels picking the labels
    for the protocol settings

    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    data_y[data_y == 24] = 0
    data_y[data_y == 12] = 8
    data_y[data_y == 13] = 9
    data_y[data_y == 16] = 10
    data_y[data_y == 17] = 11

    return data_y


def del_labels(data_t, data_x, data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function deletes the nonrelevant labels

    18 ->

    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    idy = np.where(data_y == 0)[0]
    labels_delete = idy

    idy = np.where(data_y == 8)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 9)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 10)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 11)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 18)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 19)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 20)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    return np.delete(data_t, labels_delete, 0), np.delete(data_x, labels_delete, 0), np.delete(data_y, labels_delete, 0)


def downsampling(data_t, data_x, data_y):
    """Recordings are downsamplied to 30Hz, as in the Opportunity dataset

    :param data_t: numpy integer array
        time array
    :param data_x: numpy integer array
        sensor recordings
    :param data_y: numpy integer array
        labels
    :return: numpy integer array
        Downsampled input
    """

    idx = np.arange(0, data_t.shape[0], 3)

    return data_t[idx], data_x[idx], data_y[idx]


def process_dataset_file(data):
    """Function defined as a pipeline to process individual Pamap2 files

    :param data: numpy integer matrix
        channel data: samples in rows and sensor channels in columns
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into samples-channel measurements (x) and labels (y)
    """

    # Data is divided in time, sensor data and labels
    data_t, data_x, data_y = divide_x_y(data)

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    # nonrelevant labels are deleted
    data_t, data_x, data_y = del_labels(data_t, data_x, data_y)

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    # Labels are adjusted
    data_y = adjust_idx_labels(data_y)
    data_y = data_y.astype(int)

    if data_x.shape[0] != 0:
        HR_no_NaN = complete_HR(data_x[:, 0])
        data_x[:, 0] = HR_no_NaN

        data_x[np.isnan(data_x)] = 0

        # data_x = normalize(data_x)

    else:
        data_x = data_x
        data_y = data_y
        data_t = data_t

        print("SIZE OF THE SEQUENCE IS CERO")

    data_t, data_x, data_y = downsampling(data_t, data_x, data_y)

    return data_x, data_y


def load_data_files(data_files):
    data_x = np.empty((0, NUM_FEATURES))
    data_y = np.empty((0))

    for filename in data_files:
        try:
            # data = np.loadtxt(BytesIO(zipped_dataset.read(filename)))
            data = np.loadtxt(filename)
            print('... file {0}'.format(filename))
            x, y = process_dataset_file(data)
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
        except KeyError:
            print('ERROR: Did not find {0} in zip file'.format(filename))

    return data_x, data_y


def load_data_pamap2(domain_idx):
    import os
    saved_filename = data_root + '/PAMAP2_Dataset/pamap2_domain_' + domain_idx + '_processed.data'
    if os.path.isfile(saved_filename):
        data = np.load(saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
        return X, y, d
    else:
        dataset = data_root + '/PAMAP2_Dataset/Protocol/'
        # File names of the files defining the PAMAP2 data.
        PAMAP2_DATA_FILES = ['subject101.dat',  # 0
                             # 'PAMAP2_Dataset/Optional/subject101.dat',  # 1
                             'subject102.dat',  # 2
                             'subject103.dat',  # 3
                             'subject104.dat',  # 4
                             'subject107.dat',  # 5
                             'subject108.dat',  # 6
                             # 'PAMAP2_Dataset/Optional/subject108.dat',  # 7
                             'subject109.dat',  # 8
                             # 'PAMAP2_Dataset/Optional/subject109.dat',  # 9
                             'subject105.dat',  # 10
                             # 'PAMAP2_Dataset/Optional/subject105.dat',  # 11
                             'subject106.dat',  # 12
                             # 'PAMAP2_Dataset/Optional/subject106.dat',  # 13
                             ]

        X = np.empty((0, NUM_FEATURES))
        y = np.empty((0))
        d = np.empty((0))

        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        cur_domain_files = [dataset + a for a in PAMAP2_DATA_FILES if a[7:10] == domain_idx]
        # for filename in PAMAP2_DATA_FILES:
        # if counter_files <= 9:
        # Train partition
        try:
            print('Train... file {0}'.format(cur_domain_files))
            X, y = load_data_files(cur_domain_files)
            d = np.full(y.shape, (int(domain_idx[-1]) - 1) % 100, dtype=int)
        except KeyError:
            print('ERROR: Did not find {0} in zip file'.format(cur_domain_files))

        obj = [(X, y, d)]
        # f = file(os.path.join(target_filename), 'wb')
        f = open(os.path.join(saved_filename), 'wb')

        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
        return X, y, d


def prep_domains_pamap_random(args, SLIDING_WINDOW_LEN=170, SLIDING_WINDOW_STEP=32):
    source_domain_list = ['101', '102', '103', '104', '105', '106', '107', '108', '109']
    source_position_domain_list = ['Hand', 'Chest', 'Ankle']
    # source_domain_list.remove(args.target_domain)
    # source_position_domain_list.remove(args.target_position_domain)
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, ratio = [], 0, 0.0
    # source_loaders = []
    for index, source_domain in enumerate(source_domain_list):
        # print('source_domain:', source_domain)
        x_train, y_train, domain = load_data_pamap2(source_domain)
        # (557963, 113), (557963,), (118750, 113), (118750, )
        x_win, y_win, d_win = opp_sliding_window_w_d(x_train, y_train, domain, SLIDING_WINDOW_LEN,
                                                                      SLIDING_WINDOW_STEP)

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    print("x_win_all: {}, y_win_all: {}, d_win_all: {}".format(x_win_all.shape, y_win_all.shape, d_win_all.shape))
    x_win_train, x_win_test, y_win_train, \
    y_win_test, d_win_train, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all,
                                                              split_ratio=args.split_ratio)

    print("x_win_train: {}, x_win_test: {}, y_win_train: {}, y_win_test: {}, "
          "d_win_train: {}, d_win_test: {}"
          .format(x_win_train.shape, x_win_test.shape, y_win_train.shape,
                  y_win_test.shape, d_win_train.shape, d_win_test.shape))
    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights),
                                                             replacement=True)

    train_set_r = data_loader_pamap2(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False,
                                drop_last=True, sampler=sampler, num_workers=workers, pin_memory=pin_memory)

    test_set_r = data_loader_pamap2(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False,
                               num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], test_loader_r


def prep_domains_pamap_random_fine(args, SLIDING_WINDOW_LEN=170, SLIDING_WINDOW_STEP=32):
    source_domain_list = ['101', '102', '103', '104', '105', '106', '107', '108', '109']
    source_position_domain_list = ['Hand', 'Chest', 'Ankle']
    # source_domain_list.remove(args.target_domain)
    # source_position_domain_list.remove(args.target_position_domain)
    workers = args_contains(args, 'workers', 2)
    pin_memory = args_contains(args, 'pin_memory', True)

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, ratio = [], 0, 0.0
    # source_loaders = []
    for index, source_domain in enumerate(source_domain_list):
        # print('source_domain:', source_domain)
        x_train, y_train, domain = load_data_pamap2(source_domain)
        # (557963, 113), (557963,), (118750, 113), (118750, )
        x_win, y_win, d_win = opp_sliding_window_w_d(x_train, y_train, domain, SLIDING_WINDOW_LEN,
                                                                      SLIDING_WINDOW_STEP)

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_fine_split(x_win_all, y_win_all, d_win_all,
                                                              split_ratio=args.ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)

    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights),
                                                             replacement=True)

    train_set_r = data_loader_pamap2(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False,
                                drop_last=True, sampler=sampler, num_workers=workers, pin_memory=pin_memory)
    val_set_r = data_loader_pamap2(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    test_set_r = data_loader_pamap2(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.test_batch_size, shuffle=False,
                               num_workers=workers, pin_memory=pin_memory)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_pamap2(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP):
    if args.cases == 'random':
        return prep_domains_pamap_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'random_fine':
        return prep_domains_pamap_random_fine(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'
