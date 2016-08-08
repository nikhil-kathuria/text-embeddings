
import json
import yaml
import timeit


import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("data_path", "data/", "Directory containing tha data")
flags.DEFINE_string("learing_rate", ".005", "Learning rate for the optimizer")
flags.DEFINE_string("l1_features", "600", "Number of Units in hidden layer")
flags.DEFINE_string("h1_units", "600", "Number of Units in hidden layer")
flags.DEFINE_string("num_class", "10", "Number of distinct clases")
FLAGS = flags.FLAGS


class Params:
    def __init__(self):
        # Assign option values
        self.data_path = FLAGS.data_path
        self.learning_rate = FLAGS.learning_rate
        self.l1_features = FLAGS.l1_features
        self.h1_units = FLAGS.h1_units
        self.num_class = FLAGS.num_class

        # Load the data
        self._train, self._label_train = self.loadmat('train.txt')
        self._test, self._label_test = self.loadmat('test.txt')


    def loadmat(self, fname):
        fpath = self.data_path + fname
        labels = []
        mat = []

        fobj = open(fpath, 'r')

        for line in fobj:
            arr = line.split()
            row = [0] * (len(arr) -1 )
            labels.append(arr[0].split(','))

            for col in range(1, len(arr)):
                tup = arr[col].split(":")
                row[int(tup[0])] = float(tup[1])

            mat.append(row)
        fobj.close()
        return mat, labels


class MultiNN:
    def __init__(self, params):
        self.params = params
        self.build_graph()
        tf.initialize_all_variables()

    def build_graph(self):
        ## Input Layer and Labels
        x = tf.placeholder(tf.float32, shape=[None, self.params.l1_features])
        y = tf.placeholder(tf.float32, shape=[None, self.params.num_class])

        ## Hidden Layer
        W_h1 = tf.Variable(tf.random_normal([self.params.l1_features, self.params.h1_units]))
        b_1 = tf.Variable(tf.random_normal([self.params.h1_units]))
        h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_1)

        ## Output Layer
        W_out = tf.Variable(tf.random_normal([self.params.h1_units, self.params.num_class]))
        b_out = tf.Variable(tf.random_normal([self.params.num_class]))
        y_ = tf.nn.softmax(tf.matmul(h1, W_out) + b_out)


    def opt_loss(self, logits, labels):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels, name="loss_func")

        opt = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)


    def encodeLabels(self):
        pass


    def run(self):
        # my_config to be passed as argument in tf.session(config=my_config)
        my_config = tf.ConfigProto()
        #my_config.log_device_placement=True
        my_config.allow_soft_placement=True

        with tf.Graph().as_default(), tf.Session(config=my_config) as session:
            pass











