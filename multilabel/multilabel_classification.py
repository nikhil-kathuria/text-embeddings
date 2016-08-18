
import json
import yaml
import timeit
import os


import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("data_path", "data/", "Directory containing tha data")
flags.DEFINE_string("learning_rate", ".005", "Learning rate for the optimizer")
flags.DEFINE_string("l1_features", "600", "Number of Units in hidden layer")
flags.DEFINE_string("h1_units", "600", "Number of Units in hidden layer")
flags.DEFINE_string("num_class", "10", "Number of distinct classes")
flags.DEFINE_string("save_path", ".", "Path to save model components")
flags.DEFINE_string("result_path", ".", "Path to for model results")
FLAGS = flags.FLAGS


class Params:
    def __init__(self):
        # Assign option values
        self.data_path = FLAGS.data_path
        self.learning_rate = FLAGS.learning_rate
        self.l1_features = int(FLAGS.l1_features)
        self.h1_units = int(FLAGS.h1_units)
        self.num_class = int(FLAGS.num_class)
        self.save_path = FLAGS.save_path
        self.result_path = FLAGS.result_path

        # Load the data
        self._train, self._label_train = self.loadmat('train.txt')
        self._test, self._label_test = self.loadmat('test.txt')

        # Replace the num features with actual features from data
        self.l1_features = len(self._train[0])


    def loadmat(self, fname):
        fpath = self.data_path + fname
        labels = []
        mat = []

        fobj = open(fpath, 'r')

        for line in fobj:
            arr = line.split()
            last = arr[len(arr) -1].split(":")[0]
            length = int(last) + 1
            row = np.zeros(length)
            labels.append(arr[0].split(','))
            #del arr[0]

            for col in range(1, len(arr)):
                tup = arr[col].split(":")
                #try:
                row[int(tup[0])] = float(tup[1])
                    # print(str(tup[0]) + " " + str(tup[1]))
                #except:
                    #print(str(tup[0]) + " " + str(tup[1]))
                    #exit()

            mat.append(row)
        fobj.close()
        return mat, labels


class MultiNN:
    def __init__(self, params):
        self.params = params
        self.build_graph()
        self.losslist = list()
        tf.initialize_all_variables()

    def forward_pass(self):
        ## Input Layer and Labels
        self.x = tf.placeholder(tf.float32, shape=[None, self.params.l1_features])
        self.y = tf.placeholder(tf.float32, shape=[None, self.params.num_class])

        ## Hidden Layer
        W_h1 = tf.Variable(tf.random_normal([self.params.l1_features, self.params.h1_units]))
        b_1 = tf.Variable(tf.random_normal([self.params.h1_units]))
        h1 = tf.nn.sigmoid(tf.matmul(self.x, W_h1) + b_1)

        ## Output Layer
        W_out = tf.Variable(tf.random_normal([self.params.h1_units, self.params.num_class]))
        b_out = tf.Variable(tf.random_normal([self.params.num_class]))
        self.y_ = tf.nn.softmax(tf.matmul(h1, W_out) + b_out)




    def build_graph(self):
        self.forward_pass()
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(self.y, self.y_, name="loss_func")

        tf.scalar_summary("Cross_Entropy_Loss", self.loss)

        self.train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()


    def encodeLabels(self, multi):
        rows = len(multi)
        labels = np.zeros((rows, self.params.num_class))

        for row in range(rows):
            for col in multi[row]:
                labels[row][int(col)] = 1

        return labels


def run():
    itr = 0
    # my_config to be passed as argument in tf.session(config=my_config)
    my_config = tf.ConfigProto()
    #my_config.log_device_placement=True
    my_config.allow_soft_placement=True

    with tf.Graph().as_default(), tf.Session(config=my_config) as session:
        params = Params()
        mnn = MultiNN(params)
        mnn._writer = tf.train.SummaryWriter(params.result_path + "Train", session.graph)

        while(itr < 1000):
            itr += 1
            _, loss = session.run([mnn.train, mnn.loss],
                                  feed_dict={mnn.x : params._train,
                                             mnn.y : mnn.encodeLabels(params._label_train)})
            mnn.losslist.append(loss)
            #print("Loss at iteration " + str(itr) + " " + str(loss))


        mnn.saver.save(session,
                       os.path.join(params.save_path, "model.ckpt"),
                       global_step=itr)



if __name__ == '__main__':
    run()














