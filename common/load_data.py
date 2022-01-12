# -*- coding: UTF-8 -*-

import numpy as np
from config.cfg import path, cfg
from pytorch_transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained(path['roberta_path'])


def load_data(path):
    data_X, y = [], []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            line = line.split('\t')
            data_X.append(' '.join(line[1:]))

            for i in range(1, 6):
                if str(i) in line[0]:
                    y.append(i - 1)
    return data_X, y


def get_random_sample_ids(length, K):
    import random
    ids_list = []
    for i in range(length):
        ids_list.append(i)
    ids = random.sample(ids_list, K)
    return ids


def generate_template(data_X_1, data_X_2, data_y_1, data_y_2, is_train=False):
    '''
    :param data_X_1:  target
    :param data_X_2:  train set sentence
    :return:
    '''
    data_X = []
    data_y = []
    CLS = '<s>'
    SEP = '</s>'
    MASK = '<mask>'

    neg_set_X, neg_set_y = [], []
    pos_set_X, pos_set_y = [], []
    for i in range(len(data_X_1)):
        for j in range(len(data_X_2)):
            template = cfg['template']

            template = template.replace('[X1]', CLS + ' ' + data_X_1[i] + ' ' + SEP)
            template = template.replace('[X2]', SEP + ' ' + data_X_2[j] + ' ' + SEP)
            template = template.replace('[MASK]', MASK)
            data_X.append(template)

            if data_y_1[i] == data_y_2[j]:
                data_y.append(1)
                if is_train:
                    pos_set_X.append(template)
                    pos_set_y.append(1)
            else:
                data_y.append(0)
                if is_train:
                    neg_set_X.append(template)
                    neg_set_y.append(0)

    if is_train:
        data_X = pos_set_X
        data_y = pos_set_y
        neg_ids = get_random_sample_ids(len(neg_set_X), len(pos_set_y))
        for j in neg_ids:
            data_X.append(neg_set_X[j])
            data_y.append(neg_set_y[j])

    return data_X, data_y


def data_split(data_X, data_y, K=8, Kt=8):
    train_ids = get_random_sample_ids(len(data_X), K)
    train_X, train_y = [], []
    test_all_X, test_all_y = [], []
    for i in range(len(data_X)):
        if i in train_ids:
            train_X.append(data_X[i])
            train_y.append(data_y[i])
        else:
            test_all_X.append(data_X[i])
            test_all_y.append(data_y[i])

    test_ids = get_random_sample_ids(len(test_all_X), Kt)
    test_X, test_y = [], []
    for i in range(len(test_all_X)):
        if i in test_ids:
            test_X.append(test_all_X[i])
            test_y.append(test_all_y[i])

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


def data_split_all(data_X, data_y, label_size, K=8, Kt=8):
    data_set_X, data_set_y = {}, {}
    for i in range(label_size):
        data_set_X[i] = []
        data_set_y[i] = []
    # print(data_set_X)

    for i in range(len(data_y)):
        data_set_X[data_y[i]].append(data_X[i])
        data_set_y[data_y[i]].append(data_y[i])

    train_X, train_y, test_X, test_y = [], [], [], []
    for i in range(label_size):
        train_X_t, train_y_t, test_X_t, test_y_t = data_split(data_set_X[i], data_set_y[i], K, Kt)

        if len(train_X) == 0:
            train_X, train_y, test_X, test_y = train_X_t, train_y_t, test_X_t, test_y_t
        else:
            train_X = np.hstack([train_X, train_X_t])
            train_y = np.hstack([train_y, train_y_t])
            test_X = np.hstack([test_X, test_X_t])
            test_y = np.hstack([test_y, test_y_t])

    return train_X, train_y, test_X, test_y


def data_split_balance(data_X, data_y, pos_id, num):
    '''
    正例比负例少， 随机选
    :param data_X:
    :param data_y:
    :return:
    '''

    pos_set_X, pos_set_y = [], []
    neg_set_X, neg_set_y = [], []
    pos_type = -1
    for i in range(len(data_X)):
        if data_y[i] == pos_id:
            pos_type = i // cfg['K']
            break

    for i in range(len(data_X)):
        if data_y[i] == pos_id:
            pos_set_X.append(data_X[i])
            pos_set_y.append(data_y[i])
        else:
            if abs(i//cfg['K'] - pos_type) > 0:
                neg_set_X.append(data_X[i])
                neg_set_y.append(data_y[i])

    pos_set_ids = get_random_sample_ids(len(pos_set_X), num)
    neg_set_ids = get_random_sample_ids(len(neg_set_X), num)

    data_X, data_y = [], []

    for i in pos_set_ids:
        data_X.append(pos_set_X[i])
        data_y.append(pos_set_y[i])

    for i in neg_set_ids:
        data_X.append(neg_set_X[i])
        data_y.append(neg_set_y[i])

    return data_X, data_y







