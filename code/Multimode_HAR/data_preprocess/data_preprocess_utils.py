import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
# from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# @dataclass
# class Params:
#     x: float
#     y: float
#     z: float


def train_test_val_split(x_win_all, y_win_all, d_win_all, split_ratio=0.2):
    # split all data into train and test
    x_win_train, x_win_test, y_win_train, y_win_test, d_win_train, d_win_test = \
        train_test_split(x_win_all, y_win_all, d_win_all, test_size=split_ratio, random_state=0)

    # split train into train and validation with the same ratio
    # x_win_train, x_win_val, y_win_train, y_win_val, d_win_train, d_win_val = \
    #     train_test_split(x_win_train, y_win_train, d_win_train, test_size=split_ratio, random_state=0)
    #
    # return x_win_train, x_win_val, x_win_test, \
    #        y_win_train, y_win_val, y_win_test, \
    #        d_win_train, d_win_val, d_win_test

    # no val_data
    return x_win_train, x_win_test, y_win_train, y_win_test, d_win_train, d_win_test


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


def get_sample_weights(y, weights):
    '''
    to assign weights to each sample
    '''
    label_unique = np.unique(y)
    sample_weights = []
    for val in y:
        idx = np.where(label_unique == val)
        sample_weights.append(weights[idx])
    return sample_weights


def cut_overlap(X_train):
    """
    对训练集数据求均值和标准差前先将窗口数据的重叠去掉
    :param X_train: 训练集的窗口数据 三维 ndarray
    :return long_x: 展平后的没有重叠的二维样本数据 ndarray
    """
    cut = int(X_train.shape[1] // 2)  # 50%的重叠率  cut = 200 // 2 = 100
    long_x = X_train[:, -cut:, :]  # 取每个窗口数据的第二维的后半部分 (7167, 100, 46)
    X = np.expand_dims(X_train[0, :cut, :], axis=0)  # 第一个窗口数据的前半部分需要合并 (1, 100, 46)
    long_x = np.concatenate((X, long_x), axis=0)  # 得到的是整个训练数据的展平的数据 (7168, 100, 46)
    long_x = long_x.reshape((long_x.shape[0] * long_x.shape[1], long_x.shape[2]))  # 转换成原始样本数据 (716800, 46)

    s = StandardScaler()
    # s = MinMaxScaler()
    s.fit(long_x)
    return s


def normalize(s, x):
    """
    对训练集数据和测试集数据进行标准化
    :param s: 标准化器
    :param x: 需要标准化的数据
    :return: 标准化后的数据
    """
    flat_x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
    flat_x = s.transform(flat_x)  # 将训练集中的窗口数据标准化，使用的是原始的展平后的训练集数据求的均值和标准差
    normal_x = flat_x.reshape(x.shape)  # 将展平的训练集数据转换成原来的窗口数据
    return normal_x


# def normalize(x):
#     """Normalizes all sensor channels by mean substraction,
#     dividing by the standard deviation and by 2.
#
#     :param x: numpy integer matrix
#         Sensor data
#     :return:
#         Normalized sensor data
#     """
#
#     x = np.array(x, dtype=np.float32)
#     m = np.mean(x, axis=0)
#     x -= m
#     std = np.std(x, axis=0)
#     std += 0.000001
#
#     x /= std
#     return x


def opp_sliding_window_w_d(data_x, data_y, d, ws, ss):  # window size, step size
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    data_d = np.asarray([[i[-1]] for i in sliding_window(d, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8), data_d.reshape(len(data_d)).astype(np.uint8)


def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError( \
            'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError( \
            'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape), str(ws)))
    # 窗口数量
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    # commented by hangwei
    # dim = filter(lambda i: i != 1, dim)
    return strided.reshape(dim)


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def opp_sliding_window(data_x, data_y, ws, ss):  # window size, step size
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)
