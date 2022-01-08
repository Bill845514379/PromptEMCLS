# -*- coding: UTF-8 -*-

import numpy as np
from config.cfg import path, cfg
from common.util import delete_character, delete_word, reorder_span, reorder_words
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
                    y.append(i)
    return data_X, y

def generate_template(data_X_1, data_X_2, data_y_1, data_y_2):
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

    for i in range(len(data_X_1)):
        for j in range(len(data_X_2)):
            template = cfg['template']

            template = template.replace('[X1]', CLS + ' ' + data_X_1[i] + ' ' + SEP)
            template = template.replace('[X2]', SEP + ' ' + data_X_2[j] + ' ' + SEP)
            template = template.replace('[MASK]', MASK)

            data_X.append(template)
            if data_y_1[i] == data_y_2[j]:
                data_y.append(1)
            else:
                data_y.append(0)
    return data_X, data_y

def get_random_sample_ids(length, K):
    import random
    ids_list = []
    for i in range(length):
        ids_list.append(i)
    ids = random.sample(ids_list, K)
    return ids

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





