# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec unbatched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using
  true SGD.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import json
import yaml
import timeit

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from tensorflow.models.embedding import gen_word2vec as word2vec

env = yaml.load(open('env.yaml'))['prod']

flags = tf.app.flags

flags.DEFINE_string("save_path", env['save_path'], "Directory to write the model.")
flags.DEFINE_string("result_path", env['result_path'], "Directory to write the model.")
flags.DEFINE_string(
    "train_data", env['train_data'],
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", env['eval_data'], "Analogy questions. "
    "https://word2vec.googlecode.com/svn/trunk/questions-words.txt.")
flags.DEFINE_integer("embedding_size", env['embedding_size'], "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train",env['epochs'],
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", env['learning_rate'], "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", env['negative_samples'],
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", env['batch_size'],
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", env['concurrent_steps'],
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", env['window_size'],
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", env['min_count'],
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", env['sub_sample'],
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

FLAGS = flags.FLAGS


class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.

    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # Where to write out summaries.
    self.save_path = FLAGS.save_path

    # Where to write out summaries.
    self.result_path = FLAGS.result_path

    # The text file for eval.
    self.eval_data = FLAGS.eval_data


class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []

    self.build_graph()
    #print(self._id2word)
    #print(self._word2id['the'])
    #print(self._word2id['UNK'])
    #print(self._word2id['of'])

    self.save_vocab()


  def dump_word2idx(self, path):
    with open(os.path.join(path, "word2idx.json"), 'w') as outfile:
        json.dump(self._word2id, indent=1, separators=(',', ': '), fp=outfile)



  def build_graph(self):
    """Build the model graph."""
    opts = self._options

    # The training data. A text file.
    (words, counts, words_per_epoch, current_epoch, total_words_processed,
     examples, labels) = word2vec.skipgram(filename=opts.train_data,
                                           batch_size=opts.batch_size,
                                           window_size=opts.window_size,
                                           min_count=opts.min_count,
                                           subsample=opts.subsample)
    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    opts.vocab_size = len(opts.vocab_words)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)

    self._id2word = opts.vocab_words
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i

    # Declare all variables we need.
    # Input words embedding: [vocab_size, emb_dim]
    w_in = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size,
             opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
        name="w_in")

    # Global step: scalar, i.e., shape [].
    w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")

    # Global step: []
    global_step = tf.Variable(0, name="global_step")

    # Linear learning rate decay.
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001,
        1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

    # Training nodes.
    inc = global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      train = word2vec.neg_train(w_in,
                                 w_out,
                                 examples,
                                 labels,
                                 lr,
                                 vocab_count=opts.vocab_counts.tolist(),
                                 num_negative_samples=opts.num_samples)

    self._w_in = w_in
    self.w_out = w_out
    self._examples = examples
    self._labels = labels
    self._lr = lr
    self._train = train
    self.step = global_step
    self._epoch = current_epoch
    self._words = total_words_processed

    # Properly initialize all variables and save the information
    tf.initialize_all_variables().run()
    self.saver = tf.train.Saver()

  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with open(os.path.join(opts.result_path, "vocab.txt"), "w") as f:
      for i in xrange(opts.vocab_size):
        f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_words[i]),
                             opts.vocab_counts[i]))

  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    while True:
      _, epoch = self._session.run([self._train, self._epoch])
      if epoch != initial_epoch:
        break

  def train(self):
    """Train the model."""
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    workers = []
    for _ in xrange(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    last_words, last_time = initial_words, time.time()
    while True:
      time.sleep(.01)  # Reports our progress once a while.
      (epoch, step, words,
       lr) = self._session.run([self._epoch, self.step, self._words, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      # print (epoch, step, lr, rate)
      print("Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" % (epoch, step,
                                                                    lr, rate))
      sys.stdout.flush()
      if epoch != initial_epoch:
        break

    for t in workers:
      t.join()


def dump_embed(embeddings, name, path):
  np.savetxt(os.path.join(path, name), embeddings, fmt='%10.8f')


def main(_):
  start = timeit.default_timer()
  """Train a word2vec model."""
  if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
    print("--train_data --eval_data and --save_path must be specified.")
    sys.exit(1)
  opts = Options()

  with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    model = Word2Vec(opts, session)
    model.dump_word2idx(opts.result_path)
    for _ in xrange(opts.epochs_to_train):
      model.train()  # Process one epoch

    # Do not eaval now
    # model.eval()  # Eval analogies.
    # Perform a final save.
    model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"),
                     global_step=model.step)

    if FLAGS.interactive:
      pass

    # Eval and dump the input and output embeddings
    embeddings = model._w_in.eval()
    out_embeddings = model.w_out.eval()
    dump_embed(embeddings, "embeddings.txt", opts.result_path)
    dump_embed(out_embeddings, "out_embeddings.txt", opts.result_path)
    end = timeit.default_timer()
    print("Total time " + str((end - start) / 60) + " mins")


if __name__ == "__main__":
  tf.app.run()
