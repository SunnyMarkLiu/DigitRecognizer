#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 16-10-30 下午4:37
"""
import numpy as np
from dataset import load_save_datas as lsd
import matrix_io


def extend_training_datas():
    """
    extend training datas, expand fourth. 42000 -> 168000
    """
    old_features, old_labels = lsd.load_train_data('../../dataset/train.csv')

    # extend the training datas
    extend_features = []
    extend_labels = []
    print 'extend training datas, expand fourth. 42000 -> 84000'
    for i in xrange(len(old_features)):
        feature = old_features[i]
        label = old_labels[i]
        feature = np.mat(feature)
        feature = feature.reshape(28, 28)
        for m in range(0, 5, 4):
            for n in range(0, 5, 4):
                f_temp = feature[n:28 - 4 + n, m:28 - 4 + m]
                f_temp = f_temp.reshape(1, 24 * 24)
                f_temp = f_temp.tolist()[0]
                extend_features.append(f_temp)
                extend_labels.append(label)

        print 'train:', i, np.shape(extend_features), "-", np.shape(extend_labels)

    matrix_io.save(extend_features, 'extend_train_features.pkl')
    matrix_io.save(extend_labels, 'extend_train_labels.pkl')
    print 'training datas count:', len(old_features), '-> extend training datas:', len(extend_features)


def extend_test_datas():
    """
    extend test datas, expand fourth. 28000 -> 112000
    """
    old_test_features = lsd.load_test_data('../../dataset/test.csv')

    # extend the test datas
    extend_features = []
    print 'extend test datas, expand fourth. 28000 -> 112000'
    for i in xrange(len(old_test_features)):
        feature = old_test_features[i]
        feature = np.mat(feature)
        feature = feature.reshape(28, 28)
        for m in range(0, 5, 4):
            for n in range(0, 5, 4):
                f_temp = feature[n:28 - 4 + n, m:28 - 4 + m]
                f_temp = f_temp.reshape(1, 24 * 24)
                f_temp = f_temp.tolist()[0]
                extend_features.append(f_temp)

        print 'test:', i, np.shape(extend_features)

    matrix_io.save(extend_features, 'extend_test_features.pkl')
    print 'test datas count:', len(old_test_features), '-> extend test datas:', len(extend_features)


def load_extend_training_datas():
    extend_features_dict = matrix_io.load('extend_train_features.pkl', float)
    extend_labels_dict = matrix_io.load('extend_train_labels.pkl', np.int32)

    extend_features = extend_features_dict['M']
    extend_labels = extend_labels_dict['M']

    features_mat = np.mat(extend_features) / 255.0
    labels_mat = np.zeros([len(extend_labels), 10])
    for i in xrange(len(extend_labels)):
        labels_mat[i, extend_labels[i]] = 1
    return features_mat, labels_mat


def load_extend_test_datas():
    extend_features_dict = matrix_io.load('extend_test_features.pkl', float)
    extend_features = extend_features_dict['M']
    features_mat = np.mat(extend_features) / 255.0
    return features_mat

if __name__ == '__main__':
    print 'extend data...'
    extend_training_datas()
    extend_test_datas()
    print 'extend data done!'
