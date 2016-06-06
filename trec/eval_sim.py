import json
import numpy as np
import sys
from scipy.spatial.distance import cosine


def loadmat(name):
    full = np.loadtxt(name, dtype=float)
    return full


def load_word2idx(name):
    with open(name) as out:
        return json.load(out)


def compTriplet(w1, w2, w3):
    embed = loadmat('embedding.txt')
    vocab = load_word2idx('word2idx.json')

    try:
        sim12 = 1 - cosine(embed[vocab[w1], :], embed[vocab[w2], :])
        sim13 = 1 - cosine(embed[vocab[w1], :], embed[vocab[w3], :])
    except KeyError:
        print "Word %s is not present in Vocablury" % sys.exc_value
        exit()

    print "Similarity between " + w1 + " and " + w2 + " --> " + str(sim12)
    print "Similarity between " + w1 + " and " + w3 + " --> " + str(sim13)


def make_idx2word(word2idx):
    idx2word = {word2idx[word]: word for word in word2idx}
    return idx2word


def cosine_sim(word, topk=100):
    embed = loadmat('embedding.txt')
    word2idx = load_word2idx('word2idx.json')
    idx2word = make_idx2word(word2idx)

    # Get the index of word and the corresponding vector
    try:
        index = word2idx[word]
    except KeyError:
        print "Word %s is not present in Vocablury" % sys.exc_value
        exit()
    wordvec = embed[index, :]

    # Get the distance and delete the entry at index
    cosdis = np.empty(len(embed))
    for itr in range(len(embed)):
        # Similarity = 1 - distance
        cosdis[itr] = 1 - cosine(wordvec, embed[itr:itr + 1])

    zipped = zip(range(len(embed)), cosdis)
    del zipped[index]
    zipped.sort(key=lambda t: t[1], reverse=True)

    topn = min(int(topk), len(zipped))

    print "Similarity order for " + word
    for tup in range(topn):
        print idx2word[zipped[tup][0]] + " " + str(zipped[tup][1])


def check_args():
    if len(sys.argv) == 2:
        cosine_sim(sys.argv[1])
    elif len(sys.argv) == 3:
        cosine_sim(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        compTriplet(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python eval_simp.py WORD [TOPK]")
        print("Usage: python eval_simp.py BASEWORD WORD1 WORD2")


if __name__ == '__main__':
    check_args()
