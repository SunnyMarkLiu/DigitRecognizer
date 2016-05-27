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
        data = np.array(row[1:])
        train_digit_datas.append(data)

    return train_digit_datas, train_digit_labels


def load_test_data():
    """
    加载训练数据
    :return:
    """
    test = pandas.read_csv('../dataset/test.csv')
    test_digit_datas = []
    for index, row in test.iterrows():
        data = np.array(row[0:])
        test_digit_datas.append(data)

    return test_digit_datas


def plot_count_digit_img(digit_datas, count=100):
    """
    绘制100个数据
    :return:
    """
    digit_datas = np.matrix(digit_datas)
    random_index = np.arange(0, count)
    for i in random_index:
        plt.subplot(10, 10, i + 1)  # 假设100个
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整subplot的间隔
        data = digit_datas[i].reshape((28, 28))
        plt.imshow(data, cmap=plt.get_cmap('gray'))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def knn_classify(test_digit_datas, train_digit_datas, train_digit_labels, k):
    """
    knn算法预测分类函数
    :param test_digit_datas:list
    :param train_digit_datas:list
    :param train_digit_labels:
    :param k:
    :return:
    """
    test_digit_datas = np.matrix(test_digit_datas)
    # 将 test_digit_datas 扩展成训练数据集相同shape的矩阵，便于距离的计算
    m_train = len(train_digit_datas)
    m_test = len(test_digit_datas)

    predict_labels = []
    for i in range(0, m_test):
        test_digit = test_digit_datas[i]
        test_digit = np.tile(test_digit, (m_train, 1))

        # 计算测试样本与训练数据集的距离
        diffMat = test_digit - train_digit_datas
        distances = np.sqrt(np.sum(np.square(diffMat), axis=1))
        distances = np.array(np.reshape(distances, (1, m_train)))[0]
        # 对距离进行排序，获得排序后的下标
        sorted_indexs = distances.argsort()
        labels_count = {}
        # 选出前K个距离测试数据最近的训练数据
        for j in range(0, k):
            vote_label = train_digit_labels[sorted_indexs[j]]
            # 记录K个label中出现的次数，进行下一步的vote表决
            labels_count[int(vote_label)] = labels_count.get(int(vote_label), 0) + 1

        # 选出 labels_count 中出现次数最多的 label
        sorted_labels_list = sorted(labels_count.iteritems(),
                                    key=lambda item: item[1],
                                    reverse=True)
        predict = sorted_labels_list[0][0]
        predict_labels.append(predict)
        print i, 'predict label:', predict

    return predict_labels


def save_predicts(predicts_label, filename):
    """
    保存预测结果到 csv 文件
    :return:
    """
    predicts_label.insert(0, 10)
    df = pandas.DataFrame(predicts_label)
    df.to_csv(filename)

if __name__ == '__main__':
    print 'loading training datas...'
    train_datas, train_labels = load_train_data()
    # plot_count_digit_img(train_datas)
    print 'loading test datas...'
    test_datas = load_test_data()
    print 'run KNN...'
    predicts = knn_classify(test_datas, train_datas, train_labels, 2)
    # plot_count_digit_img(test_datas)
    print 'predict labels:'
    print predicts
    save_predicts(predicts, '../dataset/sample_submission.csv')
