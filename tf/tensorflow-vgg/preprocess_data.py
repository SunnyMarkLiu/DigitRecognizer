#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
对数据预处理，28x28 -> 224x224

@author: MarkLiu
@time  : 16-11-5 上午10:36
"""
import numpy as np
from PIL import Image
import h5py
import pandas as pd
from properties import scale_image_height, scale_image_width, \
    pre_imgage_height, pre_imgage_width, VGG_MEAN


def scale_train_test_datas():
    print 'scale train datas...'
    data = pd.read_csv('../../dataset/train.csv')
    images = data.iloc[:, 1:].values
    images = images.astype(np.float)
    image_vgg = np.ndarray(shape=(images.shape[0], scale_image_height, scale_image_width, 3),
                           dtype=np.float32)
    images = images.reshape(images.shape[0], pre_imgage_height, pre_imgage_width)
    labels_flat = data[[0]].values.ravel()
    for j in range(images.shape[0]):
        pil_im = Image.fromarray(images[j])
        im_resize = pil_im.resize((scale_image_height, scale_image_width), Image.ANTIALIAS)
        im = np.array(im_resize.convert('RGB'), dtype=np.float32)
        im[:, :, 0] -= VGG_MEAN[0]
        im[:, :, 1] -= VGG_MEAN[1]
        im[:, :, 2] -= VGG_MEAN[2]
        # 'RGB'->'BGR'
        im = im[:, :, ::-1]
        image_vgg[j, :, :, :] = im
        if j % 1000 == 0:
            print 'process:%d/42000' % j
    print 'image_vgg shape:', image_vgg.shape

    labels_mat = np.zeros([len(labels_flat), 10])
    for i in xrange(len(labels_flat)):
        labels_mat[i, labels_flat[i]] = 1

    try:
        with h5py.File('precess_training_sets.h5', 'w') as f:
            f.create_dataset('train_images', data=image_vgg)
            f.create_dataset('train_labels', data=labels_mat)
            f.close()
    except Exception as e:
        print('Unable to save images:', e)

    print 'scale train datas done!'

    print 'scale test datas...'
    data = pd.read_csv('../../dataset/test.csv')
    images = data.iloc[:, 0:].values
    images = images.astype(np.float)
    image_vgg = np.ndarray(shape=(images.shape[0], scale_image_height, scale_image_width, 3), dtype=np.float32)
    images = images.reshape(images.shape[0], pre_imgage_height, pre_imgage_width)
    for j in range(images.shape[0]):
        pil_im = Image.fromarray(images[j])
        im_resize = pil_im.resize((scale_image_height, scale_image_width), Image.ANTIALIAS)
        im = np.array(im_resize.convert('RGB'), dtype=np.float32)
        im[:, :, 0] -= VGG_MEAN[0]
        im[:, :, 1] -= VGG_MEAN[1]
        im[:, :, 2] -= VGG_MEAN[2]
        # 'RGB'->'BGR'
        im = im[:, :, ::-1]
        image_vgg[j, :, :, :] = im
        if j % 1000 == 0:
            print 'process:%d/28000' % j
    print 'image_vgg shape:', image_vgg.shape

    try:
        with h5py.File('precess_test_sets.h5', 'w') as f:
            f.create_dataset('test_images', data=image_vgg)
            f.close()
    except Exception as e:
        print('Unable to save images:', e)

    print 'scale test datas done!'


def load_scaled_training_datas():
    f = h5py.File('precess_training_sets.h5', 'r')
    train_set_data = f['train_images'][:]
    train_set_labels = f['train_labels'][:]
    train_set_data /= 255.0
    return train_set_data, train_set_labels


def load_scaled_test_datas():
    f = h5py.File('precess_test_sets.h5', 'r')
    test_set_data = f['test_images'][:]
    test_set_data /= 255.0
    return test_set_data


if __name__ == '__main__':
    print 'scale data...'
    scale_train_test_datas()
    print 'scale data done!'
