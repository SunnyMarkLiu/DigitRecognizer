#!D:\Miniconda2 python
# encoding: utf-8

"""
@author: MarkLiu
@file: sklearn_knn.py
@time: 2016/5/27 14:27
"""
import load_save_datas as lsd
from sklearn import neighbors
import numpy as np


def sklearn_knn_classify(test_digit_datas, train_digit_datas, train_digit_labels):
    """

    :return:
    """
    m_test = len(test_digit_datas)

    knn = neighbors.KNeighborsClassifier()  # 取得knn分类器
    knn.fit(train_digit_datas, train_digit_labels)
    predict_labels = []
    for i in range(0, m_test):
        predict = knn.predict(np.matrix(test_digit_datas[i]))
        predict = int(predict)
        print i, 'predict label:', predict
        predict_labels.append(predict)

    return predict_labels


if __name__ == '__main__':
    print 'loading training datas...'
    train_datas, train_labels = lsd.load_train_data('../dataset/train.csv')
    # plot_count_digit_img(train_datas)
    print 'loading test datas...'
    test_datas = lsd.load_test_data('../dataset/test.csv')
    print 'run KNN...'
    predicts = sklearn_knn_classify(test_datas, train_datas, train_labels)
    print 'predicts:'
    print predicts
    lsd.save_predicts(predicts, 'sklearn_predicit.csv')
