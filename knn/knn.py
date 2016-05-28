#!D:\Miniconda2 python
# encoding: utf-8

"""
@author: MarkLiu
The slowest!  Average Accuracy: 0.968
@file: knn.py
@time: 2016/5/26 21:10
"""

import numpy as np
from sklearn.decomposition import PCA

from dataset import load_save_datas as lsd


# import plot_digit


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
    train_digit_datas = np.matrix(train_digit_datas)

    print 'before pca train_digit_datas: ', np.shape(train_digit_datas)
    # PCA降维
    pca = PCA(n_components=0.9, whiten=True)
    train_digit_datas = pca.fit_transform(train_digit_datas)
    print 'after pca train_digit_datas: ', np.shape(train_digit_datas)
    test_digit_datas = pca.transform(test_digit_datas)

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


if __name__ == '__main__':
    print 'loading training datas...'
    train_datas, train_labels = lsd.load_train_data('../dataset/train.csv')
    # plot_digit.plot_count_digit_img(train_datas)
    print 'loading test datas...'
    test_datas = lsd.load_test_data('../dataset/test.csv')
    print 'run KNN...'
    predicts = knn_classify(test_datas, train_datas, train_labels, 2)
    # plot_digit.plot_count_digit_img(test_datas)
    print 'predict labels:'
    print predicts
    lsd.save_predicts(predicts, '../dataset/sample_submission.csv')
