import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from scipy.io import loadmat
from sklearn.model_selection import KFold
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score
import torch
import pandas as pd
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def processdata(path, fold):
    data = np.load(path)
    nums, numv = np.shape(data)
    print(nums)
    fold_num = int(nums / fold)

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, : numv - 1], data[:, numv - 1:]

    KF = KFold(n_splits=fold, shuffle=True)

    for train_index, test_index in KF.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        X_val = X_train[:fold_num, :]
        Y_val = Y_train[:fold_num, :]
        X_train = X_train[fold_num:, :]
        Y_train = Y_train[fold_num:, :]

        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)  # 归一化
        X_val = min_max_scaler.fit_transform(X_val)  # 归一化
        X_test = min_max_scaler.fit_transform(X_test)  # 归一化

        XY = np.hstack((X_train, Y_train))
        sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def intra_subject_processdata(path, fold, batch, batch_size):
    data = np.load(path)
    # data = np.load('../../../data/SEED/SEED_DE.npy')
    # label = np.load('../../../data/SEED/SEED_DE_label.npy')
    # data = data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3]*data.shape[4]))
    # label = label.reshape((label.shape[0] * label.shape[1], label.shape[2], label.shape[3]))
    # print(data.shape)
    # print(label.shape)
    # data = np.concatenate((data, label), axis=2)
    domain_size, sample_size, feature_size = np.shape(data)
    label = [0, 58, 116, 171, 226, 272, 320, 379, 433, 495, 558, 616, 674, 732, 791, 841]
    # label = [0, 48, 107, 161, 223, 286, 344, 402, 460, 519, 569, 627, 685, 740, 795, 841]
    # fold_list = [9, 10, 11, 12, 13, 14]
    # fold_list = list(range(0, 6))
    fold_list = list(range(9, 15))
    # fold_list = [9, 10, 11]

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, : feature_size - 1], data[:, :, feature_size - 1:]
    data_15 = []

    sum_time = 3
    use_time = [0, 1, 2]
    for i in range(1, len(label)):
        if i < len(label):
            pre_inx = label[i-1]
            now_inx = label[i]
            data_temp = data[:, pre_inx:now_inx, :]
        else:
            pre_inx = label[i-1]
            data_temp = data[:, pre_inx:, :]
        [domain_size, sample_size, feature_size] = data_temp.shape
        res = np.empty((domain_size // sum_time, sample_size * len(use_time), feature_size))
        for i in range(domain_size // sum_time):
            inx = [i * sum_time + t for t in use_time]
            temp = np.array(data_temp[inx])
            temp = temp.reshape(-1, temp.shape[-1])
            res[i] = temp
        data_15.append(res)

    all_inx = list(range(15))
    for i in range(fold):
        test_inx = fold_list
        train_inx = [index for index in all_inx if index not in test_inx]

        test_data = [data_15[index][i] for index in test_inx]
        test_data = np.concatenate(test_data)
        train_data = [data_15[index][i] for index in train_inx]
        train_data = np.concatenate(train_data)
        print(train_data.shape)
        print(test_data.shape)

        X_train, Y_train = train_data[:, :-1], train_data[:, -1:]
        X_test, Y_test = test_data[:, :-1], test_data[:, -1:]

        # 随机选择n行
        if batch:
            n = batch_size*2  # 选择的行数
        else:
            n = X_train.shape[0]//(fold-1)
        random_indices = np.random.choice(X_train.shape[0], n, replace=False)  # 随机不重复选择行索引

        # 归一化
        d = np.concatenate([X_train, X_test], axis=0)
        # min_max_scaler = preprocessing.RobustScaler()
        min_max_scaler = preprocessing.MinMaxScaler()
        d = min_max_scaler.fit_transform(d)  # 归一化
        # X_train = min_max_scaler.fit_transform(X_train)  # 归一化
        # X_test = min_max_scaler.fit_transform(X_test)  # 归一化
        X_train = d[:X_train.shape[0]]
        X_test = d[X_train.shape[0]:]

        # 创建两个新的数组
        X_val = X_train[random_indices]
        Y_val = Y_train[random_indices]

        XY = np.hstack((X_train, Y_train))
        sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def intra_subject_processdata_V(path, fold, batch, batch_size):
    data = np.load(path)
    domain_size, sample_size, feature_size = np.shape(data)

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, : feature_size - 1], data[:, :, feature_size - 1:]
    data_15 = []

    all_inx = list(range(domain_size))
    for i in range(fold):
        d = data[i]
        train_data = d[:1199]
        test_data = d[1199:]
        print(train_data.shape)
        print(test_data.shape)

        X_train, Y_train = train_data[:, :-1], train_data[:, -1:]
        X_test, Y_test = test_data[:, :-1], test_data[:, -1:]

        # 随机选择n行
        if batch:
            n = batch_size*2  # 选择的行数
        else:
            n = X_train.shape[0]//(fold-1)
        random_indices = np.random.choice(X_train.shape[0], n, replace=False)  # 随机不重复选择行索引

        # 归一化
        d = np.concatenate([X_train, X_test], axis=0)
        # min_max_scaler = preprocessing.RobustScaler()
        min_max_scaler = preprocessing.MinMaxScaler()
        d = min_max_scaler.fit_transform(d)  # 归一化
        # X_train = min_max_scaler.fit_transform(X_train)  # 归一化
        # X_test = min_max_scaler.fit_transform(X_test)  # 归一化
        X_train = d[:X_train.shape[0]]
        X_test = d[X_train.shape[0]:]

        # 创建两个新的数组
        X_val = X_train[random_indices]
        Y_val = Y_train[random_indices]

        XY = np.hstack((X_train, Y_train))
        sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def cross_subject_multimodal_processdata(path, fold, batch, batch_size):
    data = np.load(path)
    domain_size, sample_size, feature_size = np.shape(data)
    label = [0, 58, 116, 171, 226, 272, 320, 379, 433, 495, 558, 616, 674, 732, 791, 841]
    fold_list = list(range(9, 15))

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, : feature_size - 1], data[:, :, feature_size - 1:]
    data_15 = []

    for i in range(1, len(label)):
        if i < len(label):
            pre_inx = label[i-1]
            now_inx = label[i]
            data_temp = data[:, pre_inx:now_inx, :]
        else:
            pre_inx = label[i-1]
            data_temp = data[:, pre_inx:, :]
        [domain_size, sample_size, feature_size] = data_temp.shape
        res = np.empty((domain_size // 3, sample_size * 3, feature_size))
        for i in range(domain_size // 3):
            temp = np.array(data_temp[i * 3: (i + 1) * 3])
            temp = temp.reshape(-1, temp.shape[-1])
            res[i] = temp
        data_15.append(res)

    all_inx = list(range(15))
    for i in range(fold):
        test_inx = fold_list
        train_inx = [index for index in all_inx if index not in test_inx]

        test_sample = i
        test_data = [data_15[index][test_sample] for index in test_inx]
        test_data = np.concatenate(test_data)
        train_sample = [index for index in list(range(fold)) if index != test_sample]
        train_data = []
        for sample in train_sample:
            temp = [data_15[index][sample] for index in train_inx]
            train_data.append(temp)
        train_data = np.concatenate(train_data)

        X_train, Y_train = train_data[:, :-1], train_data[:, -1:]
        X_test, Y_test = test_data[:, :-1], test_data[:, -1:]

        # 随机选择n行
        if batch:
            n = batch_size  # 选择的行数
        else:
            n = X_train.shape[0]//(fold-1)
        random_indices = np.random.choice(X_train.shape[0], n, replace=False)  # 随机不重复选择行索引

        # 创建两个新的数组
        X_val = X_train[random_indices]
        X_train = np.delete(X_train, random_indices, axis=0)
        Y_val = Y_train[random_indices]
        Y_train = np.delete(Y_train, random_indices, axis=0)

        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)  # 归一化
        X_val = min_max_scaler.fit_transform(X_val)  # 归一化
        X_test = min_max_scaler.fit_transform(X_test)  # 归一化

        XY = np.hstack((X_train, Y_train))
        sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def cross_subject_processdata_V(path, fold, batch, batch_size):
    data = np.load(path)

    domain_size, sample_size, feature_size = np.shape(data)
    fold_num = int(domain_size / fold)
    modal = [310, 33]

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, : 310], data[:, :, -1:]
    # X, Y = data[:, :, : feature_size - 1], data[:, :, feature_size - 1:]

    # KF = KFold(n_splits=fold)
    # for train_index, test_index in KF.split(X):
    all_inx = list(range(fold))
    for i in range(fold):
        test_index = i
        train_index = [index for index in all_inx if index != test_index]

        X_train, X_test = X[train_index], X[test_index]
        # print(X_train.shape, X_test.shape)
        Y_train, Y_test = Y[train_index], Y[test_index]
        # print(Y_train.shape, Y_test.shape)

        X_train = X_train.reshape(-1, X_train.shape[-1])
        Y_train = Y_train.reshape(-1, Y_train.shape[-1])
        X_test = X_test.reshape(-1, X_test.shape[-1])
        Y_test = Y_test.reshape(-1, Y_test.shape[-1])

        # 随机选择n行
        if batch:
            n = batch_size  # 选择的行数
        else:
            n = X_train.shape[0]//(fold-1)
        random_indices = np.random.choice(X_train.shape[0], n, replace=False)  # 随机不重复选择行索引

        # 创建两个新的数组
        X_val = X_train[random_indices]
        Y_val = Y_train[random_indices]
        X_train = np.delete(X_train, random_indices, axis=0)
        Y_train = np.delete(Y_train, random_indices, axis=0)
        # random_indices = np.random.choice(X_train.shape[0], 800, replace=False)
        # X_train = X_train[random_indices]
        # Y_train = Y_train[random_indices]

        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)  # 归一化
        X_val = min_max_scaler.fit_transform(X_val)  # 归一化
        X_test = min_max_scaler.fit_transform(X_test)  # 归一化

        XY = np.hstack((X_train, Y_train))
        sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list

def cross_subject_processdata(path, fold, batch, batch_size):
    data = np.load(path)
    sum_time = 3
    use_time = [0, 1]
    # use_time = [0]
    [domain_size, sample_size, feature_size] = data.shape
    data_12 = np.empty((domain_size // sum_time, sample_size * len(use_time), feature_size))
    for i in range(domain_size // sum_time):
        inx = [i * sum_time + t for t in use_time]
        temp = np.array(data[inx])
        temp = temp.reshape(-1, temp.shape[-1])
        data_12[i] = temp
    data = data_12

    domain_size, sample_size, feature_size = np.shape(data)
    fold_num = int(domain_size / fold)
    modal = [310, 33]

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, : 310], data[:, :, -1:]
    # X, Y = data[:, :, : feature_size - 1], data[:, :, feature_size - 1:]

    # KF = KFold(n_splits=fold)
    # for train_index, test_index in KF.split(X):
    all_inx = list(range(12))
    for i in range(fold):
        test_index = i
        train_index = [index for index in all_inx if index != test_index]

        X_train, X_test = X[train_index], X[test_index]
        # print(X_train.shape, X_test.shape)
        Y_train, Y_test = Y[train_index], Y[test_index]
        # print(Y_train.shape, Y_test.shape)

        X_train = X_train.reshape(-1, X_train.shape[-1])
        Y_train = Y_train.reshape(-1, Y_train.shape[-1])
        X_test = X_test.reshape(-1, X_test.shape[-1])
        Y_test = Y_test.reshape(-1, Y_test.shape[-1])

        # 随机选择n行
        if batch:
            n = batch_size  # 选择的行数
        else:
            n = X_train.shape[0]//(fold-1)
        random_indices = np.random.choice(X_train.shape[0], n, replace=False)  # 随机不重复选择行索引

        # 创建两个新的数组
        X_val = X_train[random_indices]
        Y_val = Y_train[random_indices]
        X_train = np.delete(X_train, random_indices, axis=0)
        Y_train = np.delete(Y_train, random_indices, axis=0)
        # random_indices = np.random.choice(X_train.shape[0], 800, replace=False)
        # X_train = X_train[random_indices]
        # Y_train = Y_train[random_indices]

        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)  # 归一化
        X_val = min_max_scaler.fit_transform(X_val)  # 归一化
        X_test = min_max_scaler.fit_transform(X_test)  # 归一化

        XY = np.hstack((X_train, Y_train))
        sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def Two_cross_processdata(path, fold, batch, batch_size):
    data = np.load(path)
    sum_time = 3
    use_time = [0, 1]
    # use_time = [0]
    [domain_size, sample_size, feature_size] = data.shape
    data_12 = np.empty((domain_size // sum_time, sample_size * len(use_time), feature_size))
    for i in range(domain_size // sum_time):
        inx = [i * sum_time + t for t in use_time]
        temp = np.array(data[inx])
        temp = temp.reshape(-1, temp.shape[-1])
        data_12[i] = temp
    data = data_12

    domain_size, sample_size, feature_size = np.shape(data)
    fold_num = int(domain_size / fold)
    modal = [310, 33]

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, : -1], data[:, :, -1:]
    # X, Y = data[:, :, : feature_size - 1], data[:, :, feature_size - 1:]

    # KF = KFold(n_splits=fold)
    # for train_index, test_index in KF.split(X):
    all_inx = list(range(12))
    for i in range(fold):
        test_index = i
        train_index = [index for index in all_inx if index != test_index]

        X_train, X_test = X[train_index], X[test_index]
        # print(X_train.shape, X_test.shape)
        Y_train, Y_test = Y[train_index], Y[test_index]
        # print(Y_train.shape, Y_test.shape)

        X_train = X_train.reshape(-1, X_train.shape[-1])
        Y_train = Y_train.reshape(-1, Y_train.shape[-1])
        X_test = X_test.reshape(-1, X_test.shape[-1])
        Y_test = Y_test.reshape(-1, Y_test.shape[-1])

        # 随机选择n行
        if batch:
            n = batch_size  # 选择的行数
        else:
            n = X_train.shape[0]//(fold-1)
        random_indices = np.random.choice(X_train.shape[0], n, replace=False)  # 随机不重复选择行索引

        # 创建两个新的数组
        X_val = X_train[random_indices]
        Y_val = Y_train[random_indices]
        X_train = np.delete(X_train, random_indices, axis=0)
        Y_train = np.delete(Y_train, random_indices, axis=0)
        # random_indices = np.random.choice(X_train.shape[0], 800, replace=False)
        # X_train = X_train[random_indices]
        # Y_train = Y_train[random_indices]

        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)  # 归一化
        X_val = min_max_scaler.fit_transform(X_val)  # 归一化
        X_test = min_max_scaler.fit_transform(X_test)  # 归一化

        XY = np.hstack((X_train, Y_train))
        sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def multi_domain_cross_subject_processdata(path, fold, batch, batch_size):
    data = np.load(path)
    sum_time = 3
    use_time = [0, 1]
    # use_time = [0]
    [domain_size, sample_size, feature_size] = data.shape
    data_12 = np.empty((domain_size // sum_time, sample_size * len(use_time), feature_size))
    for i in range(domain_size // sum_time):
        inx = [i * sum_time + t for t in use_time]
        temp = np.array(data[inx])
        temp = temp.reshape(-1, temp.shape[-1])
        data_12[i] = temp
    data = data_12

    domain_size, sample_size, feature_size = np.shape(data)
    fold_num = int(domain_size / fold)
    modal = [310, 33]

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, : 310], data[:, :, -1:]
    # X, Y = data[:, :, : feature_size - 1], data[:, :, feature_size - 1:]

    # KF = KFold(n_splits=fold)
    # for train_index, test_index in KF.split(X):
    all_inx = list(range(12))
    for i in range(fold):
        test_index = i
        train_index = [index for index in all_inx if index != test_index]

        X_train, X_test = X[train_index], X[test_index]
        # print(X_train.shape, X_test.shape)
        Y_train, Y_test = Y[train_index], Y[test_index]
        # print(Y_train.shape, Y_test.shape)

        # 随机选择n行
        if batch:
            n = batch_size  # 选择的行数
        else:
            n = X_train.shape[0]//(fold-1)
        random_indices = np.random.choice(X_train.shape[1], n, replace=False)  # 随机不重复选择行索引

        # 创建两个新的数组
        X_val = X_train[:, random_indices]
        Y_val = Y_train[:, random_indices]
        X_train = np.delete(X_train, random_indices, axis=1)
        Y_train = np.delete(Y_train, random_indices, axis=1)

        min_max_scaler = preprocessing.MinMaxScaler()
        for inx in range(X_train.shape[0]):
            X_train[inx] = min_max_scaler.fit_transform(X_train[inx])  # 归一化
            X_val[inx] = min_max_scaler.fit_transform(X_val[inx])  # 归一化
        X_test = min_max_scaler.fit_transform(X_test)  # 归一化

        X_train = np.transpose(X_train, (1, 0, 2))
        X_val = np.transpose(X_val, (1, 0, 2))
        Y_train = np.transpose(Y_train, (1, 0, 2))
        Y_val = np.transpose(Y_val, (1, 0, 2))

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def My_Model_Special_processdata1(path, fold, batch, batch_size):
    data = np.load(path)
    domain_size, sample_size, feature_size = np.shape(data)
    label = [0, 58, 116, 171, 226, 272, 320, 379, 433, 495, 558, 616, 674, 732, 791, 841]
    fold_list = list(range(0, 6))
    # fold_list = list(range(9, 15))

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, :-1], data[:, :, -1:]
    data_15 = []

    sum_time = 3
    use_time = [0, 1, 2]
    for i in range(1, len(label)):
        if i < len(label):
            pre_inx = label[i - 1]
            now_inx = label[i]
            data_temp = data[:, pre_inx:now_inx, :]
        else:
            pre_inx = label[i - 1]
            data_temp = data[:, pre_inx:, :]
        [domain_size, sample_size, feature_size] = data_temp.shape
        res = np.empty((domain_size // sum_time, sample_size * len(use_time), feature_size))
        for i in range(domain_size // sum_time):
            inx = [i * sum_time + t for t in use_time]
            temp = np.array(data_temp[inx])
            temp = temp.reshape(-1, temp.shape[-1])
            res[i] = temp
        data_15.append(res)

    all_inx = list(range(15))
    for i in range(fold):
        test_inx = fold_list
        train_inx = [index for index in all_inx if index not in test_inx]

        test_data = [data_15[index][i] for index in test_inx]
        test_data = np.concatenate(test_data)
        train_data = [data_15[index][i] for index in train_inx]
        train_data = np.concatenate(train_data)

        X_train, Y_train = train_data[:, :-1], train_data[:, -1:]
        X_test, Y_test = test_data[:, :-1], test_data[:, -1:]

        # 归一化
        d = np.concatenate([X_train, X_test], axis=0)
        # min_max_scaler = preprocessing.RobustScaler()
        min_max_scaler = preprocessing.MinMaxScaler()
        d = min_max_scaler.fit_transform(d)  # 归一化
        # X_train = min_max_scaler.fit_transform(X_train)  # 归一化
        # X_test = min_max_scaler.fit_transform(X_test)  # 归一化
        X_train = d[:X_train.shape[0]]
        X_test = d[X_train.shape[0]:]

        X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1])
        Y_train = Y_train.reshape(1, Y_train.shape[0])

        def split_data_by_label_max(x, y):
            unique_labels, label_counts = np.unique(y, return_counts=True)

            # 初始化分组后的数据
            grouped_x = []
            grouped_y = []

            for label, count in zip(unique_labels, label_counts):
                # 找到对应标签的索引
                indices = np.where(y[0, :] == label)
                temp_x = x[:, indices]
                temp_y = y[:, indices]
                max_num = np.max(label_counts) // x.shape[0]
                min_num = np.min(label_counts) // x.shape[0]
                if count < max_num:
                    random_inx = np.random.randint(0, count, size=(max_num - count))
                    temp_x = np.append(temp_x, temp_x[:, :, random_inx, :], axis=2)
                    temp_y = np.append(temp_y, temp_y[:, :, random_inx], axis=2)
                grouped_x.append(temp_x)
                grouped_y.append(temp_y)
            grouped_x = np.concatenate(grouped_x, axis=1)
            grouped_y = np.concatenate(grouped_y, axis=1)

            return grouped_x, grouped_y

        def split_data_by_label_min(x, y):
            unique_labels, label_counts = np.unique(y, return_counts=True)

            # 初始化分组后的数据
            grouped_x = []
            grouped_y = []

            for label, count in zip(unique_labels, label_counts):
                # 找到对应标签的索引
                indices = np.where(y[0, :] == label)
                temp_x = x[:, indices]
                max_num = np.max(label_counts) // x.shape[0]
                min_num = np.min(label_counts) // x.shape[0]
                temp_x = temp_x[:, :, :min_num, :]
                grouped_x.append(temp_x)
                temp_y = y[:, indices]
                temp_y = temp_y[:, :, :min_num]
                grouped_y.append(temp_y)
            grouped_x = np.concatenate(grouped_x, axis=1)
            grouped_y = np.concatenate(grouped_y, axis=1)

            return grouped_x, grouped_y
        grouped_x, grouped_y = split_data_by_label_max(X_train, Y_train)
        X_train = grouped_x.reshape(-1, grouped_x.shape[-2], grouped_x.shape[-1])
        Y_train = grouped_y.reshape(-1, grouped_y.shape[-1])
        Y_train = np.expand_dims(Y_train, axis=-1)

        # 随机选择n行
        if batch:
            n = batch_size  # 选择的行数
        else:
            n = X_train.shape[0] // (fold - 1)
        random_indices = np.random.choice(X_train.shape[1], n, replace=False)  # 随机不重复选择行索引

        # 创建两个新的数组
        X_val = X_train[:, random_indices]
        Y_val = Y_train[:, random_indices]

        # def fit_transform_3D(min_max_scaler, data):
        #     n, m, k = data.shape
        #     data_reshape = data.reshape((n*m, k))
        #     data_reshape = min_max_scaler.fit_transform(data_reshape)
        #     data = data_reshape.reshape((n, m, k))
        #     return data
        #
        # min_max_scaler = preprocessing.MinMaxScaler()
        # # min_max_scaler = preprocessing.RobustScaler()
        # X_train = fit_transform_3D(min_max_scaler, X_train)  # 归一化
        # X_val = fit_transform_3D(min_max_scaler, X_val)  # 归一化
        # X_test = min_max_scaler.fit_transform(X_test)  # 归一化

        X_train = np.transpose(X_train, (1, 0, 2))
        X_val = np.transpose(X_val, (1, 0, 2))
        Y_train = np.transpose(Y_train, (1, 0, 2))
        Y_val = np.transpose(Y_val, (1, 0, 2))

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def My_Model_Special_processdata_V(path, fold, batch, batch_size):
    data = np.load(path)
    domain_size, sample_size, feature_size = np.shape(data)

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, : feature_size - 1], data[:, :, feature_size - 1:]
    data_15 = []

    all_inx = list(range(domain_size))
    for i in range(fold):
        d = data[i]
        train_data = d[:1199]
        test_data = d[1199:]
        print(train_data.shape)
        print(test_data.shape)

        X_train, Y_train = train_data[:, :-1], train_data[:, -1:]
        X_test, Y_test = test_data[:, :-1], test_data[:, -1:]

        # 归一化
        d = np.concatenate([X_train, X_test], axis=0)
        # min_max_scaler = preprocessing.RobustScaler()
        min_max_scaler = preprocessing.MinMaxScaler()
        d = min_max_scaler.fit_transform(d)  # 归一化
        # X_train = min_max_scaler.fit_transform(X_train)  # 归一化
        # X_test = min_max_scaler.fit_transform(X_test)  # 归一化
        X_train = d[:X_train.shape[0]]
        X_test = d[X_train.shape[0]:]

        X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1])
        Y_train = Y_train.reshape(1, Y_train.shape[0])

        def split_data_by_label_max(x, y):
            unique_labels, label_counts = np.unique(y, return_counts=True)

            # 初始化分组后的数据
            grouped_x = []
            grouped_y = []

            for label, count in zip(unique_labels, label_counts):
                # 找到对应标签的索引
                indices = np.where(y[0, :] == label)
                temp_x = x[:, indices]
                temp_y = y[:, indices]
                max_num = np.max(label_counts) // x.shape[0]
                min_num = np.min(label_counts) // x.shape[0]
                if count < max_num:
                    random_inx = np.random.randint(0, count, size=(max_num - count))
                    temp_x = np.append(temp_x, temp_x[:, :, random_inx, :], axis=2)
                    temp_y = np.append(temp_y, temp_y[:, :, random_inx], axis=2)
                grouped_x.append(temp_x)
                grouped_y.append(temp_y)
            grouped_x = np.concatenate(grouped_x, axis=1)
            grouped_y = np.concatenate(grouped_y, axis=1)

            return grouped_x, grouped_y

        def split_data_by_label_min(x, y):
            unique_labels, label_counts = np.unique(y, return_counts=True)

            # 初始化分组后的数据
            grouped_x = []
            grouped_y = []

            for label, count in zip(unique_labels, label_counts):
                # 找到对应标签的索引
                indices = np.where(y[0, :] == label)
                temp_x = x[:, indices]
                max_num = np.max(label_counts) // x.shape[0]
                min_num = np.min(label_counts) // x.shape[0]
                temp_x = temp_x[:, :, :min_num, :]
                grouped_x.append(temp_x)
                temp_y = y[:, indices]
                temp_y = temp_y[:, :, :min_num]
                grouped_y.append(temp_y)
            grouped_x = np.concatenate(grouped_x, axis=1)
            grouped_y = np.concatenate(grouped_y, axis=1)

            return grouped_x, grouped_y
        grouped_x, grouped_y = split_data_by_label_max(X_train, Y_train)
        X_train = grouped_x.reshape(-1, grouped_x.shape[-2], grouped_x.shape[-1])
        Y_train = grouped_y.reshape(-1, grouped_y.shape[-1])
        Y_train = np.expand_dims(Y_train, axis=-1)

        # 随机选择n行
        if batch:
            n = batch_size  # 选择的行数
        else:
            n = X_train.shape[0] // (fold - 1)
        random_indices = np.random.choice(X_train.shape[1], n, replace=False)  # 随机不重复选择行索引

        # 创建两个新的数组
        X_val = X_train[:, random_indices]
        Y_val = Y_train[:, random_indices]

        # def fit_transform_3D(min_max_scaler, data):
        #     n, m, k = data.shape
        #     data_reshape = data.reshape((n*m, k))
        #     data_reshape = min_max_scaler.fit_transform(data_reshape)
        #     data = data_reshape.reshape((n, m, k))
        #     return data
        #
        # min_max_scaler = preprocessing.MinMaxScaler()
        # # min_max_scaler = preprocessing.RobustScaler()
        # X_train = fit_transform_3D(min_max_scaler, X_train)  # 归一化
        # X_val = fit_transform_3D(min_max_scaler, X_val)  # 归一化
        # X_test = min_max_scaler.fit_transform(X_test)  # 归一化

        X_train = np.transpose(X_train, (1, 0, 2))
        X_val = np.transpose(X_val, (1, 0, 2))
        Y_train = np.transpose(Y_train, (1, 0, 2))
        Y_val = np.transpose(Y_val, (1, 0, 2))

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def My_Model_Special_processdata2(path, fold, batch, batch_size):
    data = np.load(path)
    sum_time = 3
    use_time = [0, 1]
    # use_time = [0]
    [domain_size, sample_size, feature_size] = data.shape
    data_12 = np.empty((domain_size // sum_time, sample_size * len(use_time), feature_size))
    for i in range(domain_size // sum_time):
        inx = [i * sum_time + t for t in use_time]
        temp = np.array(data[inx])
        temp = temp.reshape(-1, temp.shape[-1])
        data_12[i] = temp
    data = data_12

    domain_size, sample_size, feature_size = np.shape(data)
    fold_num = int(domain_size / fold)
    modal = [310, 33]

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, : 310], data[:, :, -1:]
    # X, Y = data[:, :, : feature_size - 1], data[:, :, feature_size - 1:]

    # KF = KFold(n_splits=fold)
    # for train_index, test_index in KF.split(X):
    all_inx = list(range(12))
    for i in range(fold):
        test_index = i
        train_index = [index for index in all_inx if index != test_index]

        X_train, X_test = X[train_index], X[test_index]
        # print(X_train.shape, X_test.shape)
        Y_train, Y_test = Y[train_index], Y[test_index]
        # print(Y_train.shape, Y_test.shape)

        def split_data_by_label(x, y):
            unique_labels, label_counts = np.unique(y, return_counts=True)

            # 初始化分组后的数据
            grouped_x = []
            grouped_y = []

            for label, count in zip(unique_labels, label_counts):
                # 找到对应标签的索引
                indices = np.where(y[0, :] == label)
                temp_x = x[:, indices]
                num = np.min(label_counts)//x.shape[0]
                temp_x = temp_x[:, :, :num, :]
                grouped_x.append(temp_x)
                temp_y = y[:, indices]
                temp_y = temp_y[:, :, :num]
                grouped_y.append(temp_y)
            grouped_x = np.concatenate(grouped_x, axis=1)
            grouped_y = np.concatenate(grouped_y, axis=1)

            return grouped_x, grouped_y

        # 分组
        grouped_x, grouped_y = split_data_by_label(X_train, Y_train.squeeze())
        X_train = grouped_x.reshape(-1, grouped_x.shape[-2], grouped_x.shape[-1])
        Y_train = grouped_y.reshape(-1, grouped_y.shape[-1])
        Y_train = np.expand_dims(Y_train, axis=-1)

        # 随机选择n行
        if batch:
            n = batch_size  # 选择的行数
        else:
            n = X_train.shape[0]//(fold-1)
        random_indices = np.random.choice(X_train.shape[1], n, replace=False)  # 随机不重复选择行索引

        def fit_transform_3D(min_max_scaler, data):
            n, m, k = data.shape
            data_reshape = data.reshape((n*m, k))

            # # 将numpy数组转换为Pandas DataFrame
            # df = pd.DataFrame(data_reshape[:, 0:6], columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6'])
            # # 设置图形布局
            # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
            #
            # # 绘制每个特征的直方图和核密度估计图
            # for i, name in enumerate(df.columns):
            #     row = i // 3
            #     col = i % 3
            #
            #     sns.histplot(df[name], kde=True, ax=axes[row, col], color='skyblue', bins=20)
            #     axes[row, col].set_title(f'Distribution of {col}')
            #
            # plt.tight_layout()
            # plt.show()

            data_reshape = min_max_scaler.fit_transform(data_reshape)
            data = data_reshape.reshape((n, m, k))
            return data

        min_max_scaler = preprocessing.RobustScaler()
        min_max_scaler = preprocessing.MinMaxScaler()
        for n in range(11):
            temp_eeg = X_train[n * 3:(n + 1) * 3, :310]
            X_train[n * 3:(n + 1) * 3, :310] = fit_transform_3D(min_max_scaler, temp_eeg)
        X_test = min_max_scaler.fit_transform(X_test)  # 归一化

        # 创建两个新的数组
        X_val = X_train[:, random_indices]
        Y_val = Y_train[:, random_indices]

        X_train = np.transpose(X_train, (1, 0, 2))
        X_val = np.transpose(X_val, (1, 0, 2))
        Y_train = np.transpose(Y_train, (1, 0, 2))
        Y_val = np.transpose(Y_val, (1, 0, 2))
        print(X_train.shape, Y_train.shape)

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def My_Model_Special_processdata3(path, fold, batch, batch_size):
    data = np.load(path)
    sum_time = 3
    use_time = [0, 1]
    # use_time = [0]
    [domain_size, sample_size, feature_size] = data.shape
    data_12 = np.empty((domain_size // sum_time, sample_size * len(use_time), feature_size))
    for i in range(domain_size // sum_time):
        inx = [i * sum_time + t for t in use_time]
        temp = np.array(data[inx])
        temp = temp.reshape(-1, temp.shape[-1])
        data_12[i] = temp
    data = data_12

    domain_size, sample_size, feature_size = np.shape(data)
    fold_num = int(domain_size / fold)
    modal = [310, 33]

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, :, :-1], data[:, :, -1:]
    # X, Y = data[:, :, : feature_size - 1], data[:, :, feature_size - 1:]

    # KF = KFold(n_splits=fold)
    # for train_index, test_index in KF.split(X):
    all_inx = list(range(12))
    for i in range(fold):
        test_index = i
        train_index = [index for index in all_inx if index != test_index]

        X_train, X_test = X[train_index], X[test_index]
        # print(X_train.shape, X_test.shape)
        Y_train, Y_test = Y[train_index], Y[test_index]
        # print(Y_train.shape, Y_test.shape)

        def split_data_by_label(x, y):
            unique_labels, label_counts = np.unique(y, return_counts=True)

            # 初始化分组后的数据
            grouped_x = []
            grouped_y = []

            for label, count in zip(unique_labels, label_counts):
                # 找到对应标签的索引
                indices = np.where(y[0, :] == label)
                temp_x = x[:, indices]
                num = np.min(label_counts)//x.shape[0]
                temp_x = temp_x[:, :, :num, :]
                grouped_x.append(temp_x)
                temp_y = y[:, indices]
                temp_y = temp_y[:, :, :num]
                grouped_y.append(temp_y)
            grouped_x = np.concatenate(grouped_x, axis=1)
            grouped_y = np.concatenate(grouped_y, axis=1)

            return grouped_x, grouped_y

        # 分组
        grouped_x, grouped_y = split_data_by_label(X_train, Y_train.squeeze())
        X_train = grouped_x.reshape(-1, grouped_x.shape[-2], grouped_x.shape[-1])
        Y_train = grouped_y.reshape(-1, grouped_y.shape[-1])
        Y_train = np.expand_dims(Y_train, axis=-1)

        # 随机选择n行
        if batch:
            n = batch_size  # 选择的行数
        else:
            n = X_train.shape[0]//(fold-1)
        random_indices = np.random.choice(X_train.shape[1], n, replace=False)  # 随机不重复选择行索引

        def fit_transform_3D(min_max_scaler, data):
            n, m, k = data.shape
            data_reshape = data.reshape((n*m, k))
            data_reshape = min_max_scaler.fit_transform(data_reshape)
            data = data_reshape.reshape((n, m, k))
            return data

        # min_max_scaler = preprocessing.RobustScaler()
        min_max_scaler = preprocessing.MinMaxScaler()
        for n in range(11):
            temp_eeg = X_train[n*3:(n+1)*3, :310]
            X_train[n*3:(n+1)*3, :310] = fit_transform_3D(min_max_scaler, temp_eeg)
            temp_eye = X_train[n * 3:(n + 1) * 3, 310:]
            X_train[n * 3:(n + 1) * 3, 310:] = fit_transform_3D(min_max_scaler, temp_eye)

        X_test[:, :310] = min_max_scaler.fit_transform(X_test[:, :310])  # 归一化
        X_test[:, 310:] = min_max_scaler.fit_transform(X_test[:, 310:])  # 归一化

        # 创建两个新的数组
        X_val = X_train[:, random_indices]
        Y_val = Y_train[:, random_indices]

        X_train = np.transpose(X_train, (1, 0, 2))
        X_val = np.transpose(X_val, (1, 0, 2))
        Y_train = np.transpose(Y_train, (1, 0, 2))
        Y_val = np.transpose(Y_val, (1, 0, 2))

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def getfold(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    xy_tr = list(zip(X_train, Y_train))
    xy_val = list(zip(X_val, Y_val))
    xy_te = list(zip(X_test, Y_test))

    class_list = list()
    y_list = Y_train.squeeze().tolist()
    labels = np.unique(Y_train.squeeze())
    for i in range(len(labels)):
        class_list.append(y_list.count(i))
    return xy_tr, xy_val, xy_te, class_list


def loaddata(x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list, i):
    xy_tr, xy_val, xy_te, class_list = getfold(x_train_list[i], y_train_list[i], x_val_list[i], y_val_list[i],
                                               x_test_list[i], y_test_list[i])

    batch_size = sum(class_list)
    # batch_size = 1024
    train_loader = DataLoader(
        xy_tr, batch_size=batch_size)

    val_loader = DataLoader(
        xy_val, batch_size=batch_size)

    test_loader = DataLoader(
        xy_te, batch_size=batch_size)

    return train_loader, val_loader, test_loader, class_list


def cross_subject_getfold(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    xy_tr = list(zip(X_train, Y_train))
    xy_val = list(zip(X_val, Y_val))
    xy_te = list(zip(X_test, Y_test))
    return xy_tr, xy_val, xy_te


def multi_domain_cross_subject_loaddata(x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list, i,
                           batch, batch_size=1024):
    if batch:
        xy_tr, xy_val, xy_te = cross_subject_getfold(x_train_list[i], y_train_list[i],
                                                                 x_val_list[i], y_val_list[i],
                                                                 x_test_list[i], y_test_list[i])
        sample_num = batch_size
    else:
        xy_tr, xy_val, xy_te, class_list = getfold(x_train_list[i], y_train_list[i], x_val_list[i], y_val_list[i],
                                                   x_test_list[i], y_test_list[i])
        sample_num = sum(class_list)

    train_loader = DataLoader(
        xy_tr, batch_size=sample_num, shuffle=True, drop_last=True)

    val_loader = DataLoader(
        xy_val, batch_size=sample_num, shuffle=True)

    test_loader = DataLoader(
        xy_te, batch_size=sample_num, shuffle=True)

    return train_loader, val_loader, test_loader, sample_num


def multi_domain_cross_subject_loaddata2(x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list, i,
                           batch, batch_size=1024):
    if batch:
        xy_tr, xy_val, xy_te = cross_subject_getfold(x_train_list[i], y_train_list[i],
                                                                 x_val_list[i], y_val_list[i],
                                                                 x_test_list[i], y_test_list[i])
        sample_num = batch_size
    else:
        xy_tr, xy_val, xy_te, class_list = getfold(x_train_list[i], y_train_list[i], x_val_list[i], y_val_list[i],
                                                   x_test_list[i], y_test_list[i])
        sample_num = sum(class_list)

    train_loader = DataLoader(
        xy_tr, batch_size=sample_num, shuffle=True, drop_last=True)

    val_loader = DataLoader(
        xy_val, batch_size=sample_num, shuffle=True, drop_last=True)

    test_loader = DataLoader(
        xy_te, batch_size=sample_num*3, shuffle=True, drop_last=True)

    return train_loader, val_loader, test_loader, sample_num


def cross_subject_loaddata(x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list, i,
                           batch, batch_size=1024):
    if batch:
        xy_tr, xy_val, xy_te = cross_subject_getfold(x_train_list[i], y_train_list[i],
                                                                 x_val_list[i], y_val_list[i],
                                                                 x_test_list[i], y_test_list[i])
        sample_num = batch_size
    else:
        xy_tr, xy_val, xy_te, class_list = getfold(x_train_list[i], y_train_list[i], x_val_list[i], y_val_list[i],
                                                   x_test_list[i], y_test_list[i])
        sample_num = sum(class_list)

    train_loader = DataLoader(
        xy_tr, batch_size=sample_num, shuffle=True, drop_last=True)

    val_loader = DataLoader(
        xy_val, batch_size=sample_num, shuffle=True)

    test_loader = DataLoader(
        xy_te, batch_size=sample_num, shuffle=True)

    return train_loader, val_loader, test_loader, sample_num


def list_to_np(x):
    temp = np.zeros((1, x[0].shape[1]))
    for i in range(len(x)):
        temp = np.vstack((temp, x[i].data.cpu().numpy()))
    return temp[1:, :]


def divide(x, modalities):
    x_list = []
    for i in range(len(modalities) - 1):
        index1 = modalities[i]
        index2 = modalities[i + 1]
        x_list.append(x[:, index1:index2])

    return x_list


def eval(z, z_te, svmc=0.5, average='macro'):
    z = list_to_np(z)
    z_te = list_to_np(z_te)
    x_train, y_train = z[:, :-1], z[:, -1]

    x_test, y_test = z_te[:, :-1], z_te[:, -1]

    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)

    clf = SVC(kernel='linear', C=svmc)
    clf.fit(x_train, y_train.ravel())
    acc_tr = clf.score(x_train, y_train)
    acc_te = clf.score(x_test, y_test)

    # auc = roc_auc_score(y_test, clf.decision_function(x_test))
    score = clf.decision_function(x_test)
    y_test_score = np.exp(score) / np.sum(np.exp(score), axis=1, keepdims=True)
    auc = roc_auc_score(y_test, y_test_score, average=average, multi_class='ovr')
    # 计算标准差
    std = np.std(y_test_score)

    y_true = y_test.squeeze()
    y_pre = clf.predict(x_test)
    fptn = 0
    tn = 0
    tpfn = 0
    tp = 0
    for n in range(len(y_pre)):
        if y_true[n] == 0:
            fptn = fptn + 1
            if y_pre[n] == y_true[n]:
                tn = tn + 1
        if y_true[n] == 1:
            tpfn = tpfn + 1
            if y_pre[n] == y_true[n]:
                tp = tp + 1
    if tn == 0:
        spe = 0
    else:
        spe = tn / fptn
    if tp == 0:
        sen = 0
    else:
        sen = tp / tpfn

    # 计算精确度（Precision）
    pre = precision_score(y_true, y_pre, average=average, zero_division=1)

    # 计算 F1 分数
    f1 = f1_score(y_true, y_pre, average=average)

    return acc_tr, acc_te, sen, spe, auc, pre, f1, std


def SVM_eval(z, z_te, svmc=0.5, average='macro'):
    z = list_to_np(z)
    z_te = list_to_np(z_te)
    x_train, y_train = z[:, :-1], z[:, -1]

    x_test, y_test = z_te[:, :-1], z_te[:, -1]

    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)

    clf = SVC(kernel='linear', C=svmc)
    clf.fit(x_train, y_train.ravel())

    train_score = clf.decision_function(x_train)
    test_score = clf.decision_function(x_test)
    y_train_score = np.exp(train_score) / np.sum(np.exp(train_score), axis=1, keepdims=True)
    y_test_score = np.exp(test_score) / np.sum(np.exp(test_score), axis=1, keepdims=True)
    return y_train_score, y_train, y_test_score, y_test


def cross_subject_eval(y_pred, y_true, average='macro'):
    # calculate_metrics
    # 转化为numpy
    if torch.is_tensor(y_pred):
        y_pred_label = y_pred.argmax(dim=1).cpu().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.long().cpu().numpy()
    else:
        y_pred_label = np.argmax(y_pred, axis=1)

    row_sum = y_pred.sum(axis=1)
    y_pred = y_pred / row_sum[:, np.newaxis]
    y_score = y_pred[:, y_pred_label]

    std = np.std(y_score)

    # 计算准确度（Accuracy）
    accuracy = accuracy_score(y_true, y_pred_label)

    # 计算混淆矩阵（Confusion Matrix）
    conf_matrix = confusion_matrix(y_true, y_pred_label)

    # 计算 ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred, average=average, multi_class='ovr')

    # 计算召回率（Sensitivity/Recall）
    sensitivity = recall_score(y_true, y_pred_label, average=average)

    # 计算特异性（Specificity）
    if conf_matrix[0, 0] + conf_matrix[0, 1] == 0:
        specificity = 0  # 处理分母为零的情况
    else:
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    # 计算精确度（Precision）
    pre = precision_score(y_true, y_pred_label, average=average, zero_division=1)

    # 计算 F1 分数
    f1 = f1_score(y_true, y_pred_label, average=average)

    num_classes = len(np.unique(y_true))
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    for i in range(num_classes):
        y_true_class = (y_true == i)
        y_pre_class = (y_pred_label == i)
        precision[i] = precision_score(y_true_class, y_pre_class, zero_division=1)
        recall[i] = recall_score(y_true_class, y_pre_class, zero_division=1)

    return {
        "Accuracy": accuracy,
        "Confusion Matrix": conf_matrix,
        "ROC AUC": roc_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": pre,
        "F1 Score": f1,
        "STD": std,
    }, precision, recall

