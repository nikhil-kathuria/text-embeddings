from getText import genindex, indxlist
import numpy as np
import tensorflow as tf
import math
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops


def genBatch(data, batch, window):
    global index
    assert batch % window == 0

    index = max(index, window / 2 )
    batch_size = max(0, min(batch, len(data) - index - window))

    batch = list()
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)


    counter = 0
    while (counter < batch_size):
        idx_list = list()
        for idx in range(index - window, index + window + 1):
            if idx == index:
                labels[counter] = data[idx]
            else:
                idx_list.append(data[idx])
        counter +=1
        batch.append(idx_list)
        # Update the index
        index += 1
    return batch, labels



def graphUpdate():
    global index
    graph = tf.Graph()
    with graph.as_default():

      # Input data.
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2 * window])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

      # Ops and variables pinned to the CPU because of missing GPU implementation
      with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        print(type(embed))
        print(tf.shape(embed))



        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


        # my = tf.variable(tf.nn.embedding_lookup(embeddings, train_inputs))
        tmp = tf.reduce_sum(
                      tf.nn.embedding_lookup(
                              embeddings, train_inputs))
        print(tf.shape(tmp))
        embed = functional_ops.map_fn(
              lambda wvec : tf.reduce_sum(
                      tf.nn.embedding_lookup(
                              embeddings, wvec), 0) / batch_size, train_inputs, parallel_iterations=batch_size, dtype=tf.float32)
        print(type(embed))
        print(tf.shape(embed))
        exit()




      #embed = tf.reduce_sum(tf.nn.embedding_lookup(embeddings, train_inputs), 0) / 8
      # tf.Variable.assign(nce_weights[index], tf.reduce_sum(
      #        tf.nn.embedding_lookup(embeddings, train_inputs), 0) / np.full(train_inputs, len(train_inputs) ,dtype=np.int32))

      # tf.Variable.assign(nce_weights[index], ( tf.reduce_sum(tf.nn.embedding_lookup(embeddings, train_inputs), 0) / len(train_labels)) )
      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      loss = tf.reduce_mean(
          tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                         num_sampled, vocabulary_size))

      # Construct the SGD optimizer using a learning rate of 1.0.
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
      # Normalize the embeddings
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm




    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      tf.initialize_all_variables().run()
      print("Initialized")

      average_loss = 0
      step = 0
      while(step <= steplimit ):
        batch_inputs, batch_labels = genBatch(
            sentpos, batch_size, window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        # Update

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % scheck == 0:
          if step > 0:
            average_loss /= scheck
          print("Average loss at step ", step, ": ", average_loss)
          print(index)
          average_loss = 0

          # Update the embeddings
        final_embeddings = normalized_embeddings.eval()
        exit()
        step +=1

    return final_embeddings


if __name__ == '__main__':

    wordmap, sentpos = indxlist('sam8.txt')
    vocabulary_size = len(wordmap)
    embedding_size = 32
    num_sampled = 32
    batch_size = 8
    window = 1
    scheck = 10
    steplimit = 2000


    index = window
    embeddings = graphUpdate()
    print embeddings