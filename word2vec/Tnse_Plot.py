import json
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

def loadmat(name):
    full = np.loadtxt(name, dtype=float)
    return full


def loadvocab():
    with open('vocab.json') as out:
        return json.load(out)


def loaddict():
    fobj = open('/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/dictionary.txt','r')

    mydict = set()
    for word in fobj:
        mydict.add(word.strip())
    return mydict


def validTokens(vocab, mydict):
    valid = list()
    for key in vocab.keys():
        if key in mydict:
            valid.append(vocab[key])

    return valid


def reverseVocab(vocab):
    mydict = dict()
    for key in vocab.keys():
        mydict.update({vocab[key] : key })

    return mydict


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)


def runTnse():
    embed = loadmat('Embedding.txt')
    mydict = loaddict()
    vocab = loadvocab()
    reversevocab = reverseVocab(vocab)

    valid = validTokens(vocab, mydict)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

    plot_only = 100

    low_dim_embs = tsne.fit_transform(embed[valid[:plot_only],:])

    labels = [reversevocab[itr] for itr in valid[:plot_only] ]

    assert low_dim_embs.shape

    plot_with_labels(low_dim_embs, labels)


def compTriplet(w1, w2, w3):
    embed = loadmat('Embedding.txt')
    vocab = loadvocab()

    sim12 = 1 - cosine(embed[vocab[w1],:], embed[vocab[w2],:])
    sim13 = 1 - cosine(embed[vocab[w1],:], embed[vocab[w3],:])

    print "Similarity between " + w1 +  " and " + w2 + " --> " + str(sim12)
    print "Similarity between " + w1 +  " and " + w3 + " --> " +  str(sim13)


def nearestCosine(name, topN):
    embed = loadmat('Embedding.txt')
    vocab = loadvocab()
    reversevocab = reverseVocab(vocab)

    # Get the index of word and the corresponding vector
    index = vocab[name]
    wordvec = embed[index,:]

    # print(embed.shape)
    # print(embed[0:1])
    # print(wordvec)
    # print cosine(embed[0:1], wordvec)

    # Get the distance and delete the entry at index
    cosdis = np.empty(len(embed))
    for itr in range(len(embed)):
        cosdis[itr] = cosine(wordvec, embed[itr:itr + 1])

    zipped = zip(range(len(embed)), cosdis)
    del zipped[index]
    zipped.sort(key = lambda t: t[1])

    print "Nearest words to " + name
    for tup in range(topN):
        print reversevocab[zipped[tup][0]] + " " + str(zipped[tup][1])



if __name__ == '__main__':
    # runTnse()
    #nearestCosine('hurricane',100)
    nearestCosine('rain',100)
    #nearestCosine('floods',100)
    # compTriplet('structure', 'structures', 'infrastructure')
    # compTriplet('rain', 'floods', 'hurricane')







