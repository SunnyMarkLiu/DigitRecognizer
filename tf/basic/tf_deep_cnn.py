#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
tensorflow convolutional neural network

conv/pool/relu/conv/pool/relu/fc/readout

@author: MarkLiu
@time  : 16-10-23 下午4:43
"""
import pandas
import tensorflow as tf
import numpy as np
from dataset import load_save_datas as lsd
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


class DigitsModel(object):
    """
    手写数字识别的 CNN 模型
    """

    def create_weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def create_bias_variable(self, shape, name):
        initial = tf.constant(np.random.rand(), shape=shape)
        return tf.Variable(initial, name=name)

    def create_conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def create_max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def create_model(self):
        # input features
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        # reshape features to 2d shape
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        # correct labels
        self.y_correct = tf.placeholder(tf.float32, [None, 10])

        # layer1: conv(features map: 32, 5x5, input channel: 1) + pool
        self.W_conv1 = self.create_weight_variable([5, 5, 1, 32], 'W_conv1')
        self.b_conv1 = self.create_bias_variable([32], 'b_conv1')
        self.conv1 = tf.nn.relu(self.create_conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.pool1 = self.create_max_pool_2x2(self.conv1)

        # layer1: conv(features map: 64, 5x5, input channel: 32) + pool
        self.W_conv2 = self.create_weight_variable([5, 5, 32, 64], 'W_conv2')
        self.b_conv2 = self.create_bias_variable([64], 'b_conv2')
        self.conv2 = tf.nn.relu(self.create_conv2d(self.pool1, self.W_conv2) + self.b_conv2)
        self.pool2 = self.create_max_pool_2x2(self.conv2)

        # fully-connected layer
        # connects 3136 features (7*7*64) from _pool2 to 1024 _fc nodes
        self.W_fc1 = self.create_weight_variable([7 * 7 * 64, 1024], 'W_fc1')
        self.b_fc1 = self.create_bias_variable([1024], 'b_fc1')
        self.pool2_flat = tf.reshape(self.pool2, [-1, 7 * 7 * 64])
        self.full_con = tf.nn.relu(tf.matmul(self.pool2_flat, self.W_fc1) + self.b_fc1)

        # dropout layer
        self.keep_prob = tf.placeholder(tf.float32)  # how many features to keep
        self.dropout = tf.nn.dropout(self.full_con, self.keep_prob)

        # readout layer
        # 1024 features from _fc+_dropout -> 10 outputs
        self.W_readout = self.create_weight_variable([1024, 10], 'W_readout')
        self.b_read_out = self.create_bias_variable([10], 'b_read_out')
        self.digits = tf.matmul(self.dropout, self.W_readout) + self.b_read_out
        self.read_out = tf.nn.softmax(self.digits)

    def init_training_run_op(self):
        """
        设置 session.run 的 op
        """
        # loss function
        self.loss_function = tf.nn.softmax_cross_entropy_with_logits(self.read_out, self.y_correct)

        # training op
        self.training_op = tf.train.AdamOptimizer(1e-5).minimize(self.loss_function)

        # match predicted values against the correct ones, True or False
        self.predict_matches = tf.equal(tf.argmax(self.read_out, 1), tf.argmax(self.y_correct, 1))

        # accuracy metric
        self.accuracy = tf.reduce_mean(tf.cast(self.predict_matches, tf.float32))

        # test op
        self.test_op = tf.argmax(self.read_out, 1)

    def init(self):
        self.create_model()
        self.init_training_run_op()

        init_op = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init_op)

    def train_step(self, features_batch, labels_batch):
        feed_dict = {
            self.x: features_batch,
            self.y_correct: labels_batch,
            self.keep_prob: 0.5
        }
        self.sess.run(self.training_op, feed_dict=feed_dict)

    def get_accuracy(self, features, labels):
        feed_dict = {
            self.x: features,
            self.y_correct: labels,
            self.keep_prob: 1.0
        }
        return self.sess.run(self.accuracy, feed_dict=feed_dict)

    def clarify(self, features):
        test_labels = self.sess.run(self.test_op, feed_dict={self.x: features, self.keep_prob: 1.0})
        return test_labels

    def get_layer(self, features):
        conv1layer, pool1layer, conv2layer, pool2layer = self.sess.run([self.conv1, self.pool1, self.conv2, self.pool2],
                                                                       feed_dict={self.x: features,
                                                                                  self.keep_prob: 1.0})
        print 'conv1layer: ', self.sess.run(tf.shape(conv1layer))
        print 'pool1layer: ', self.sess.run(tf.shape(pool1layer))
        print 'conv2layer: ', self.sess.run(tf.shape(conv2layer))
        print 'pool2layer: ', self.sess.run(tf.shape(pool2layer))


def load_training_datas():
    """
    load training data, use one-hot encoding
    """
    features, labels = lsd.load_train_data('../../dataset/train.csv')
    features_mat = np.mat(features)
    features_mat = np.divide(features_mat, 255.0)
    labels_mat = np.zeros([len(labels), 10])
    for i in xrange(len(labels)):
        labels_mat[i, labels[i]] = 1
    return features_mat, labels_mat


def load_test_datas():
    """
    load training data, use one-hot encoding
    """
    features = lsd.load_test_data('../../dataset/test.csv')
    features_mat = np.mat(features)
    features_mat = np.divide(features_mat, 255.0)
    return features_mat


def generate_batch(features, labels, batch_size):
    batch_indexes = np.random.random_integers(0, len(features) - 1, batch_size)
    batch_features = features[batch_indexes]
    batch_labels = labels[batch_indexes]
    return batch_features, batch_labels


print 'load training data...'
features, labels = load_training_datas()
print 'load training data...Done!'

# training params
BATCH_SIZE = 200
TRAIN_SPLIT = 0.85  # training/validation split
TRAINING_STEPS = int(len(features) * TRAIN_SPLIT / BATCH_SIZE) * 500
print 'training epochs: ', TRAINING_STEPS
# split data into training and validation sets
train_samples = int(len(features) * TRAIN_SPLIT)
train_features = features[:train_samples]
train_labels = labels[:train_samples]
validation_features = features[train_samples:]
validation_labels = labels[train_samples:]

model = DigitsModel()
model.init()

accuracy_history = []

# 测试时输出各层的结构信息
model.get_layer(train_features[:1])

for epoch in xrange(TRAINING_STEPS):

    if epoch % 100 == 0 or epoch == TRAINING_STEPS - 1:
        accuracy = model.get_accuracy(features=validation_features, labels=validation_labels)
        accuracy_history.append(accuracy)
        print 'total: ', TRAINING_STEPS, '\tstep ', epoch, '\tvalidation accuracy: ', accuracy

    batch_features, batch_labels = generate_batch(train_features, train_labels, BATCH_SIZE)
    model.train_step(batch_features, batch_labels)

# plot validation accuracy, and adjust params
fig = plt.figure()
plt.ylim(bottom=0, top=1)
plt.xlim(0, len(accuracy_history))
plt.plot(accuracy_history)
fig.savefig('accuracy_history.png', dpi=75)

# test data
test_features = load_test_datas()
test_labels = model.clarify(test_features)
test_labels = np.append([100], test_labels)
df = pandas.DataFrame(test_labels)
df.to_csv('tf_cnn_test_labels.csv', sep=',')
