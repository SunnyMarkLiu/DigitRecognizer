#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 16-11-4 下午3:18
"""
import inspect
import os

import numpy as np
import tensorflow as tf
import time

# mean value is from computing the average of each layer in the training data.
VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16(object):
    def __init__(self, vgg_pretrained_model_file=None):
        if vgg_pretrained_model_file is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, 'vgg16.npy')
            vgg_pretrained_model_file = path
            print path
        self.vgg_pretrained_model_dict = np.load(vgg_pretrained_model_file, encoding='latin1').item()
        print('npy file loaded')

    def build_model(self, rgb_images):
        """
        load vgg pre-trained model with given model file

        :param rgb_images: rgb image tensor [batch, height, width, 3] values scaled [0, 1]
        """
        print 'build pre-trained vgg model'
        start_time = time.time()
        rgb_scaled_images = rgb_images * 255.0

        # Convert RGB to BGR, for opencv issue.
        red, green, blue = tf.split(3, 3, rgb_scaled_images)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr_images = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr_images.get_shape().as_list()[1:] == [224, 224, 3]

        # 构建 vgg16 模型，各层的参数参考: https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
        self.conv1_1 = self.create_conv_layer(bgr_images, 'conv1_1')
        self.conv1_2 = self.create_conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.create_conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.create_conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.create_conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.create_conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.create_conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.create_conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.create_conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.create_conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.create_conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.create_conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.create_conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        # full-connect
        # 注意 VGG16 的最后两个全连接层在训练时添加了 dropout 层（keep_prop=0.5）
        self.fc6 = self.create_fullconnect_layer(self.pool5, 'fc6')
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu_fc6 = tf.nn.relu(self.fc6, 'relu_fc6')

        self.fc7 = self.create_fullconnect_layer(self.relu_fc6, 'fc7')
        self.relu_fc7 = tf.nn.relu(self.fc7, 'relu_fc7')

        self.fc8 = self.create_fullconnect_layer(self.relu_fc7, 'fc8')

        self.softmax = tf.nn.softmax(self.fc8, name='softmax')

        self.vgg_pretrained_model_dict = None

        print("build model finished: %ds" % (time.time() - start_time))

    def max_pool(self, inputs, name):
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def average_pool(self, inputs, name):
        return tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def create_conv_layer(self, inputs, layer_name):
        """
        构建卷积层
        """
        with tf.variable_scope(layer_name):
            conv_filter = self.get_vgg_conv_filter(layer_name)
            conv_bias = self.get_vgg_bias(layer_name)
            conv = tf.nn.conv2d(inputs, conv_filter, strides=[1, 1, 1, 1], padding='SAME', name=layer_name)
            relu = tf.nn.relu(conv + conv_bias)

            return relu

    def create_fullconnect_layer(self, inputs, layer_name):
        """
        构建全连接层
        """
        with tf.variable_scope(layer_name):
            shape = inputs.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            flaten = tf.reshape(inputs, [-1, dim])

            weights = self.get_vgg_fc_weights(layer_name)
            biases = self.get_vgg_bias(layer_name)

            fc = tf.matmul(flaten, weights) + biases
            return fc

    def get_vgg_conv_filter(self, layer_name):
        """
        获取预训练的 VGG 各卷积层的卷积核，作为模型卷积核的初始值
        """
        return tf.constant(self.vgg_pretrained_model_dict[layer_name][0], name=layer_name + '-filter')

    def get_vgg_bias(self, layer_name):
        """
        获取预训练的 VGG 各层的 bias，作为模型 bias 的初始值
        """
        return tf.constant(self.vgg_pretrained_model_dict[layer_name][1], name=layer_name + '-bias')

    def get_vgg_fc_weights(self, layer_name):
        """
        获取预训练的 VGG 全连接层的权重，作为模型全连接层权重的初始值
        """
        return tf.constant(self.vgg_pretrained_model_dict[layer_name][0], name=layer_name + '-weights')
