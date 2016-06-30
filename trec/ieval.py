import json
import numpy as np
import sys
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
from colorama import Fore
from IPython import embed


# Define helper functions first
def loadmat(name):
    full = np.loadtxt(name, dtype=float)
    return full


def load_word2idx(name):
    with open(name) as out:
        return json.load(out)


def make_idx2word(word2idx):
    idx2word = {word2idx[word]: word for word in word2idx}
    return idx2word


# Declare Global Static variables first
gpath = '/gss_gpfs_scratch/kathuria.n/data/google_news/'
myembed = loadmat('embedding.txt')
word2idx = load_word2idx('word2idx.json')
idx2word = make_idx2word(word2idx)
wv=Word2Vec.load_word2vec_format(gpath + 'GoogleNews-vectors-negative300.bin', binary=True)


# Function to display which word is closer to first word
def triplet(w1, w2, w3):
    try:
        sim12 = 1 - cosine(myembed[word2idx[w1], :], myembed[word2idx[w2], :])
        sim13 = 1 - cosine(myembed[word2idx[w1], :], myembed[word2idx[w3], :])
    except KeyError:
        print "Word %s is not present in Vocablury" % sys.exc_value
        return

    print "Similarity between " + w1 + " and " + w2 + " --> " + str(sim12)
    print "Similarity between " + w1 + " and " + w3 + " --> " + str(sim13)


# Function to get Top K similar from google news corpus
def gcosine(word, topk):
    similar = wv.most_similar(positive=["protest"],topn=100)

    for tup in similar:
        print('\t' + str(tup[0][0]) + " " + str(tup[0][1]))


# Generic function to compute cosine similarity for a word against corpus
def cosine_compute(word, topk=100):
    # Get the index of word and the corresponding vector
    try:
        index = word2idx[word]
        wordvec = myembed[index, :]
    except KeyError:
        print "Word %s is not present in Vocablury" % sys.exc_value
        return


    # Get the distance and delete the entry at index
    cosdis = np.empty(len(myembed))
    for itr in range(len(myembed)):
        # Similarity = 1 - distance
        cosdis[itr] = 1 - cosine(wordvec, myembed[itr:itr + 1])

    zipped = zip(range(len(myembed)), cosdis)
    del zipped[index]
    zipped.sort(key=lambda t: t[1], reverse=True)

    return  zipped


# Function to get Top K similar from our corpus
def tcosine(word, topk=100):
    zipped = cosine_compute(word, topk)
    topn = min(int(topk), len(zipped))

    print "Similarity order for " + word
    for tup in range(topn):
        print idx2word[zipped[tup][0]] + " " + str(zipped[tup][1])


# Function to print the comparison of google vs our corpus result
def compare(word, topk=100):
    # Check for words presence in google news corpus
    try:
        gsim = wv.most_similar(positive=[word], topk=100)
    except KeyError:
        print "Word %s is not present in Google News" % sys.exc_value
        return

    # Check for words presence in our loaded corpus
    mysim = cosine_compute(word, topk)

    # Now compare and contrast
    topn = min(int(topk), len(mysim), len(gsim))
    gmap = {}
    mymap = {}

    for itr in range(topn):
        gmap[str(gsim[itr][0])] = gsim[itr][1]
        mymap[str(mysim[itr][0])] = mysim[itr][1]


    for itr in range(topn):
        gword = str(gsim[itr][0])
        mword = str(mysim[itr][0])

        if gword in mymap:
            part1 = (Fore.GREEN + gword + " " + str(gsim[itr][1]))
        else:
            part1 = (Fore.RED + gword + " " + str(gsim[itr][1]))

        if mword in gmap:
            part2 = (Fore.GREEN + gword + " " + str(mysim[itr][1]))
        else:
            part2 = (Fore.RED + gword + " " + str(mysim[itr][1]))

        # Print the the combined line
        print('\t' + part1 + '\t||\t' + part2 + '\t')


if __name__ == '__main__':
    embed()