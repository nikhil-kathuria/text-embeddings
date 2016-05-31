from glob import glob
import json
import os


import streamcorpus
from Cleanse import *

inpath="/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec"
outpath="/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec/data"



def dump_tfmap(qmap, fname, idt):
    with open(fname, 'w') as outfile:
        json.dump(qmap, indent=idt, separators=(',', ': '), fp=outfile)


def dump_dfmap(qmap, fname):
    dfmap = {term: len(qmap[term]) for term in qmap}
    dump_tfmap(dfmap, fname, 1)


def updatemap(qmap, word, docno):
    if word in qmap:
        dmap = qmap[word]
        if docno in dmap:
            dmap[docno] += 1
        else:
            dmap[docno] = 1
    else:
        mydict = {docno : 1}
        qmap[word] = mydict


def extract(fname, corpus, out):
    qmap = dict()

    fobj = open(fname, 'w')

    for si in streamcorpus.Chunk(path=corpus):
        doctext = ""
        docno = si.doc_id

        ## get the tag-stripped text from 'clean_visible'
        if si.body.clean_visible is None:
            continue

        # Here we concatenate the words into a space-separated string. Saves us the
        # hassle of cleaning and tokenizing the clean_visible text.
        try:
            sentences = si.body.sentences["serif"]
        except KeyError:
            sentences = si.body.sentences["lingpipe"]

        # One entry per sentence. Could concatenate further if desired.
        # sentencesa = [" ".join([x.token for x in s.tokens]) for s in sentences]
        for s in sentences:
            sent = ""
            for x in s.tokens:
                # word = replaceurl(onlyenglish(x.token)).lower()
                word = replaceurl(replacemail(x.token))
                word = remove_(exceptwords(word)).lower()

                # Update the IR map
                updatemap(qmap, word, docno)

                # Form sentences and doc and finally write
                sent = sent + " " + word.strip()
            doctext = doctext + " " + sent.strip()
        fobj.write(shrinkspace(doctext) + " ")

    # Post processing
    fobj.close()

    tf_file = out + "/" + "TFmap.json"
    df_file = out + "/" + "DFmap.json"
    dump_tfmap(qmap, tf_file, 2)
    dump_dfmap(qmap, df_file)


def read():
    for filename in glob(inpath + '/*'):
        spa = filename.split('/')
        dirname = spa[len(spa) - 1].split('.')[0]

        out = outpath + "/" + dirname
        os.makedirs(out)
        fileout = out + "/" + 'data.txt'

        print("Processing -> " + filename)
        extract(fileout, filename, out)





if __name__ == '__main__':
    read()