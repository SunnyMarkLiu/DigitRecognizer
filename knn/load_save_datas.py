#!D:\Miniconda2 python
# encoding: utf-8

"""
@author: MarkLiu
@file: loaddatas.py
@time: 2016/5/27 14:25
"""
import numpy as np
import pandas


def load_train_data(filename):
    """
    加载训练数据
    :return:
    """
    train = pandas.read_csv(filename)
    train_digit_labels = []
    train_digit_datas = []
    for index, row in train.iterrows():
        key = row['label']
        train_digit_labels.append(key)
        data = np.array(row[1:])
        train_digit_datas.append(data)

    return train_digit_datas, train_digit_labels


def load_test_data(filename):
    """
    加载训练数据
    :return:
    """
    test = pandas.read_csv(filename)
    test_digit_datas = []
    for index, row in test.iterrows():
        data = np.array(row[0:])
        test_digit_datas.append(data)

    return test_digit_datas


def save_predicts(predicts_label, filename):
    """
    保存预测结果到 csv 文件
    :return:
    """
    predicts_label.insert(0, 10)
    df = pandas.DataFrame(predicts_label)
    df.to_csv(filename)
