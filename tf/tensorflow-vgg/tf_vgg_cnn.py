#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 16-11-4 下午1:24
"""
import numpy as np
import pandas
import tensorflow as tf

import preprocess_data as datasets
import vgg16_pretrained_model as vgg16
from properties import scale_image_height, scale_image_width, labels_counts


class DigitsModel(object):
    """
    DigitRecognizer model
    """

    def __init__(self):
        self.vgg16 = vgg16.Vgg16(use_fullconnect=False)

    def build_model(self):
        # input features
        self.x_image = tf.placeholder(tf.float32, shape=[None, scale_image_height, scale_image_width, 3])

        # correct labels
        self.y_correct = tf.placeholder(tf.float32, [None, labels_counts])

        # dropout layer: keep probability
        self.keep_prob = tf.placeholder(tf.float32)  # how many features to keep

        # learning_rate placeholder
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        self.vgg16.build_model(self.x_image)
        self.pool5 = self.vgg16.get_vgg_out()

        dim = 1
        shape = self.pool5.get_shape().as_list()
        for i in shape[1:]:
            dim *= i

        self.pool5_flat = tf.reshape(self.pool5, [-1, dim])

        # fully-connected layer + dropout
        self.W_fc6 = self.create_weight_variable([dim, 1024], 'W_fc6')
        self.b_fc6 = self.create_bias_variable([1024], 'b_fc6')
        self.full_con_6 = tf.nn.relu(tf.matmul(self.pool5_flat, self.W_fc6) + self.b_fc6)
        print self.full_con_6

        # self.W_fc7 = self.create_weight_variable([1024, 1024], 'W_fc7')
        # self.b_fc7 = self.create_bias_variable([1024], 'b_fc7')
        # self.full_con_7 = tf.nn.relu(tf.matmul(self.full_con_6, self.W_fc7) + self.b_fc7)
        # print self.full_con_7

        self.W_fc8 = self.create_weight_variable([1024, 256], 'W_fc8')
        self.b_fc8 = self.create_bias_variable([256], 'b_fc8')
        self.full_con_8 = tf.nn.relu(tf.matmul(self.full_con_6, self.W_fc8) + self.b_fc8)

        self.dropout = tf.nn.dropout(self.full_con_8, self.keep_prob)
        print self.full_con_8

        # readout layer
        self.W_readout = self.create_weight_variable([256, 10], 'W_readout')
        self.b_read_out = self.create_bias_variable([10], 'b_read_out')
        self.digits = tf.matmul(self.dropout, self.W_readout) + self.b_read_out
        # softmax layer
        self.read_out = tf.nn.softmax(self.digits)
        print self.read_out

    def init_training_run_op(self):
        """
        设置 session.run 的 op
        """
        # loss function
        self.loss_function = tf.nn.softmax_cross_entropy_with_logits(self.read_out, self.y_correct)
        cross_entropy = tf.reduce_mean(self.loss_function)
        tf.scalar_summary('cross entropy', cross_entropy)

        # training op
        self.training_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function)

        # match predicted values against the correct ones, True or False
        self.predict_matches = tf.equal(tf.argmax(self.read_out, 1), tf.argmax(self.y_correct, 1))

        # accuracy metric
        self.accuracy = tf.reduce_mean(tf.cast(self.predict_matches, tf.float32))
        tf.scalar_summary('accuracy', self.accuracy)
        # test op
        self.test_op = tf.argmax(self.read_out, 1)

    def init(self):
        self.build_model()
        self.init_training_run_op()
        init_op = tf.initialize_all_variables()
        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter('summaries_log' + '/train',
                                                   self.sess.graph)
        self.validate_writer = tf.train.SummaryWriter('summaries_log' + '/validate')
        self.sess.run(init_op)

    def train_step(self, features_batch, labels_batch, learning_rate):
        feed_dict = {
            self.x_image: features_batch,
            self.y_correct: labels_batch,
            self.keep_prob: 0.25,
            self.learning_rate: learning_rate
        }
        summary, _ = self.sess.run([self.merged, self.training_op], feed_dict=feed_dict)
        return summary

    def get_accuracy(self, features, labels):
        feed_dict = {
            self.x_image: features,
            self.y_correct: labels,
            self.keep_prob: 1.0
        }
        summary, accuracy = self.sess.run([self.merged, self.accuracy], feed_dict=feed_dict)
        return summary, accuracy

    def clarify(self, features):
        test_labels = self.sess.run(self.test_op, feed_dict={self.x_image: features, self.keep_prob: 1.0})
        return test_labels

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def create_weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def create_bias_variable(self, shape, name):
        initial = tf.constant(np.random.rand(), shape=shape)
        return tf.Variable(initial, name=name)

    def get_train_writer(self):
        return self.train_writer

    def get_validate_writer(self):
        return self.validate_writer


def generate_batch(features, labels, batch_size):
    batch_indexes = np.random.random_integers(0, len(features) - 1, batch_size)
    batch_features = features[batch_indexes]
    batch_labels = labels[batch_indexes]
    return batch_features, batch_labels


if __name__ == '__main__':

    model = DigitsModel()
    model.init()

    print 'load scaled training data...'
    features, labels = datasets.load_scaled_training_datas()
    print 'load scaled training data...Done!'

    print 'scaled training features:', np.shape(features), 'scaled training labels:', np.shape(labels)
    # training params
    TRAIN_BATCH_SIZE = 50
    VALIDATE_BATCH_SIZE = 100
    TRAIN_SPLIT = 0.90  # training/validation split
    TRAINING_STEPS = int(len(features) * TRAIN_SPLIT / TRAIN_BATCH_SIZE) * 50
    print 'training epochs: ', TRAINING_STEPS
    # split data into training and validation sets
    train_samples = int(len(features) * TRAIN_SPLIT)
    train_features = features[:train_samples]
    train_labels = labels[:train_samples]
    validation_features = features[train_samples:]
    validation_labels = labels[train_samples:]

    accuracy_history = []

    temp_validate_accuracy = 0
    learning_rate = 1e-4
    for epoch in xrange(TRAINING_STEPS):

        if epoch % 100 == 0 or epoch == TRAINING_STEPS - 1:
            batch_features, batch_labels = generate_batch(validation_features, validation_labels, VALIDATE_BATCH_SIZE)
            summary, accuracy = model.get_accuracy(features=batch_features, labels=batch_labels)
            accuracy_history.append(accuracy)
            model.get_validate_writer().add_summary(summary, epoch)
            print 'learning_rate:', learning_rate, 'total: ', TRAINING_STEPS, '\tstep ', epoch, '\tvalidation accuracy: ', accuracy
            if accuracy < temp_validate_accuracy and learning_rate > 1e-10:  # 如果验证集减小
                learning_rate /= 3
            temp_validate_accuracy = accuracy

        batch_features, batch_labels = generate_batch(train_features, train_labels, TRAIN_BATCH_SIZE)
        summary = model.train_step(batch_features, batch_labels, learning_rate)
        if epoch % 100 == 0 or epoch == TRAINING_STEPS - 1:
            model.get_train_writer().add_summary(summary, epoch)

    model.get_validate_writer().close()
    model.get_train_writer().close()

    # delete some data to save memorry
    del features
    del labels
    del accuracy_history

    print 'predict test datas...'
    # test data
    test_features = datasets.load_scaled_test_datas()
    test_batch_size = 100
    test_labels = np.array([], dtype=np.int32)
    for i in range(len(test_features) / test_batch_size):
        labels = model.clarify(test_features[test_batch_size * i: test_batch_size * (i + 1)])
        test_labels = np.append(test_labels, labels)

    print 'save test predict...'
    df = pandas.DataFrame({'ImageId': range(1, len(test_labels) + 1, 1), 'Label': test_labels})
    df.to_csv('tf_advance_cnn_test_labels.csv', sep=',', index=False, columns=["ImageId", "Label"])
    print 'done!'
