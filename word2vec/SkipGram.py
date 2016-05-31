from collections import Counter
import pprint
from getText import genindex, indxlist, wordArray
import numpy as np
import tensorflow as tf
import math
import json


def genBatch(data, batch, window):
    global index
    assert batch % window == 0 # Batch should contain exactly 1 more integer windows
    # The batch size cannot be greater than index + batchsize > length
    #IGNORE batch_size = max(0, min(batch, len(data) - (index + 1) - window))

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    counter = 0
    while counter < batch_size :
        for itr in range(index - window, index + window + 1):
            if itr == index:
                continue
            batch[counter] = data[index]
            labels[counter, 0] = data[itr]
            # Update the counter for within the batch
            counter +=1
        # Update the index to begin the next batch
        index += 1

    return batch, labels


def filterWords(name, min):
    wordlist = wordArray(name)
    pruned = Counter(wordlist)

    sentpos = list()
    vocab = dict()
    vocab['unk'] = 0

    for word in wordlist:
        if word not in vocab:
            vocab[word] = len(vocab)
        if pruned[word] >= min:
            sentpos.append(vocab[word])
        else:
            sentpos.append(vocab['unk'])

    return vocab, sentpos




def graphUpdate():
    global index
    graph = tf.Graph()
    with graph.as_default():

      # Input data.
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

      # Ops and variables pinned to the CPU because of missing GPU implementation
      with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

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
      while(step <= steplimit and (len(sentpos) - index) >= batch_size ):
        batch_inputs, batch_labels = genBatch(
            sentpos, batch_size, window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % scheck == 0:
          if step > 0:
            average_loss /= scheck
          # The average loss is an estimate of the loss over the last 2000 batches.
          print("Average loss at step ", step, ": Index ", index, " : ", average_loss)
          average_loss = 0

          # Compute the cosine similarity between minibatch examples and all embeddings.
        final_embeddings = normalized_embeddings.eval()
        step +=1

    return final_embeddings, nce_weights


def dumpvocab(vocab):
    with open('vocab.json', 'w') as outfile:
        json.dump(vocab, indent=1, separators=(',', ': '), fp=outfile)


if __name__ == '__main__':
    # wordmap, sentpos = indxlist('sam8.txt')
    wordmap, sentpos = filterWords('whole.txt', 3)
    vocabulary_size = len(wordmap)
    embedding_size = 64
    num_sampled = 32
    batch_size = 16
    window = 2
    scheck = 10
    steplimit = 200000000
    print len(sentpos)
    print len(wordmap)

    index = window
    embeddings, nce_weights = graphUpdate()

    # final_embeddings = np.resize(embeddings,(vocabulary_size, embedding_size))
    dumpvocab(wordmap)

    print embeddings.shape

    np.savetxt('Embedding.txt', embeddings, fmt='%10.6f')
    # np.savetxt('Weights.txt', nce_weights, fmt='%10.6f')

