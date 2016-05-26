#!D:\Miniconda2 python
# encoding: utf-8

"""
@author: MarkLiu
@file: knn.py
@time: 2016/5/26 21:10
"""

import numpy as np
import pandas
from matplotlib.pylab import plt


def load_train_data():
    """
    加载训练数据
    :return:
    """
    train = pandas.read_csv('../dataset/train.csv')
    train_digit_labels = []
    train_digit_datas = []
    for index, row in train.iterrows():
        key = row['label']
        train_digit_labels.append(key)
        data = np.array(row[1:]).reshape((28, 28))
        train_digit_datas.append(data)

    return train_digit_datas, train_digit_labels


def plot_100_digit_img(digit_datas):
    """
    绘制100个数据
    :return:
    """
    random_index = np.arange(0, 100)
    for i in random_index:
        plt.subplot(10, 10, i + 1)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整subplot的间隔
        data = digit_datas[i]
        plt.imshow(data, cmap=plt.get_cmap('gray'))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def knn_classify(test_digit_datas, train_digit_datas, train_digit_labels, k):
    """
    knn算法预测分类函数
    :param test_digit_datas:
    :param train_digit_datas:
    :param train_digit_labels:
    :param k:
    :return:
    """


if __name__ == '__main__':
    train_datas, train_labels = load_train_data()
    plot_100_digit_img(train_datas)
