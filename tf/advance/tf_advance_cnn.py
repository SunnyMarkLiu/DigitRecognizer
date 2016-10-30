#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
tensorflow advance deeper convolutional neural network
conv/conv/pool/relu/dropout/    conv/conv/pool/relu/dropout/  fc/fc/fc/dropout/ readout

@author: MarkLiu
@time  : 16-10-30 上午10:32
"""
import pandas
import tensorflow as tf
import numpy as np
from dataset import load_save_datas as lsd
import matplotlib as mpl
import matrix_io

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
        self.x = tf.placeholder(tf.float32, shape=[None, 24 * 24])
        # reshape features to 2d shape
        self.x_image = tf.reshape(self.x, [-1, 24, 24, 1])

        # correct labels
        self.y_correct = tf.placeholder(tf.float32, [None, 10])

        # dropout layer: keep probability
        self.keep_prob = tf.placeholder(tf.float32)  # how many features to keep

        # layer1: conv + conv + pool
        self.W_conv1 = self.create_weight_variable([3, 3, 1, 32], 'W_conv1')
        self.b_conv1 = self.create_bias_variable([32], 'b_conv1')
        self.conv1 = tf.nn.relu(self.create_conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.W_conv1_2 = self.create_weight_variable([3, 3, 32, 32], 'W_conv1_2')
        self.b_conv1_2 = self.create_bias_variable([32], 'b_conv1_2')
        self.conv1_2 = tf.nn.relu(self.create_conv2d(self.conv1, self.W_conv1_2) + self.b_conv1_2)
        self.pool1 = self.create_max_pool_2x2(self.conv1_2)
        # add dropout layer in hidden layer1
        self.dropout1 = tf.nn.dropout(self.pool1, self.keep_prob)

        # layer2: conv + conv + pool
        self.W_conv2 = self.create_weight_variable([3, 3, 32, 64], 'W_conv2')
        self.b_conv2 = self.create_bias_variable([64], 'b_conv2')
        self.conv2 = tf.nn.relu(self.create_conv2d(self.dropout1, self.W_conv2) + self.b_conv2)
        self.W_conv2_2 = self.create_weight_variable([3, 3, 64, 64], 'W_conv2_2')
        self.b_conv2_2 = self.create_bias_variable([64], 'b_conv2_2')
        self.conv2_2 = tf.nn.relu(self.create_conv2d(self.conv2, self.W_conv2_2) + self.b_conv2_2)
        self.pool2 = self.create_max_pool_2x2(self.conv2)
        # add dropout layer in hidden layer2
        self.dropout2 = tf.nn.dropout(self.pool2, self.keep_prob)

        # fully-connected layer
        self.W_fc1 = self.create_weight_variable([6 * 6 * 64, 1024], 'W_fc1')
        self.b_fc1 = self.create_bias_variable([1024], 'b_fc1')
        self.pool2_flat = tf.reshape(self.dropout2, [-1, 6 * 6 * 64])
        self.full_con_1 = tf.nn.relu(tf.matmul(self.pool2_flat, self.W_fc1) + self.b_fc1)

        self.W_fc2 = self.create_weight_variable([1024, 1024], 'W_fc2')
        self.b_fc2 = self.create_bias_variable([1024], 'b_fc2')
        self.full_con_2 = tf.nn.relu(tf.matmul(self.full_con_1, self.W_fc2) + self.b_fc2)

        self.W_fc3 = self.create_weight_variable([1024, 256], 'W_fc3')
        self.b_fc3 = self.create_bias_variable([256], 'b_fc3')
        self.full_con_3 = tf.nn.relu(tf.matmul(self.full_con_2, self.W_fc3) + self.b_fc3)
        self.dropout = tf.nn.dropout(self.full_con_3, self.keep_prob)

        # readout layer
        # 1024 features from _fc+_dropout -> 10 outputs
        self.W_readout = self.create_weight_variable([256, 10], 'W_readout')
        self.b_read_out = self.create_bias_variable([10], 'b_read_out')
        self.digits = tf.matmul(self.dropout, self.W_readout) + self.b_read_out
        # softmax layer
        self.read_out = tf.nn.softmax(self.digits)

    def init_training_run_op(self):
        """
        设置 session.run 的 op
        """
        # loss function
        self.loss_function = tf.nn.softmax_cross_entropy_with_logits(self.read_out, self.y_correct)

        # training op
        self.training_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss_function)

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
        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()
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

    return extend_features, extend_labels


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

    return extend_features


def load_extend_training_datas():
    extend_features_dict = matrix_io.load('extend_train_features.pkl', float)
    extend_labels_dict = matrix_io.load('extend_train_labels.pkl', float)

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


def generate_batch(features, labels, batch_size):
    batch_indexes = np.random.random_integers(0, len(features) - 1, batch_size)
    batch_features = features[batch_indexes]
    batch_labels = labels[batch_indexes]
    return batch_features, batch_labels


if __name__ == '__main__':
    features, labels = extend_training_datas()
    print 'load extended training data...'
    features, labels = load_extend_training_datas()
    print 'load extended training data...Done!'

    print 'extended training features:', np.shape(features), 'extended training labels:', np.shape(labels)
    # training params
    BATCH_SIZE = 200
    TRAIN_SPLIT = 0.85  # training/validation split
    TRAINING_STEPS = int(len(features) * TRAIN_SPLIT / BATCH_SIZE) * 100
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

    # delete some data to save memorry
    del features
    del labels
    del accuracy_history

    print 'predict test datas...'
    # test data
    test_features = load_test_datas()
    test_batch_size = 100
    test_labels = np.array([], dtype=np.int32)
    for i in range(len(test_features) / test_batch_size):
        labels = model.clarify(test_features[test_batch_size * i: test_batch_size * (i + 1)])
        test_labels = np.append(test_labels, labels)

    df = pandas.DataFrame({'ImageId': range(1, len(test_labels) + 1, 1), 'Label': test_labels})
    df.to_csv('tf_advance_cnn_test_labels.csv', sep=',', index=False, columns=["Label", "ImageId"])
