#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 16-10-30 下午4:37
"""
import numpy as np
from PIL import Image

import matrix_io
from dataset import load_save_datas as lsd


def extend_training_datas():
    old_features, old_labels = lsd.load_train_data('../../dataset/train.csv')

    # extend the training datas
    extend_features = []
    extend_labels = []
    for i in xrange(len(old_features)):
        feature = old_features[i]
        label = old_labels[i]
        pil_im = Image.fromarray(feature)
        im_resize = pil_im.resize((224, 224), Image.ANTIALIAS)
        extend_features.append(im_resize)
        extend_labels.append(label)

        print 'train:', i, np.shape(extend_features), "-", np.shape(extend_labels)

    matrix_io.save(extend_features, 'extend_train_features.pkl')
    matrix_io.save(extend_labels, 'extend_train_labels.pkl')


def extend_test_datas():
    """
    extend test datas, expand fourth. 28000 -> 112000
    """
    old_test_features = lsd.load_test_data('../../dataset/test.csv')

    # extend the test datas
    extend_features = []
    for i in xrange(len(old_test_features)):
        feature = old_test_features[i]
        feature = np.mat(feature)
        resized_img = Image.fromarray(feature)
        resized_img.resize((224, 224))
        extend_features.append(resized_img)
        print 'test:', i, np.shape(extend_features)

    matrix_io.save(extend_features, 'extend_test_features.pkl')


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
