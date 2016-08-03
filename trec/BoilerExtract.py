from glob import glob
import json
import os

from Cleanse import *
from boilerpipe.extract import Extractor
from collections import Counter
import streamcorpus

inpath = "/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec/data"
outpath = "/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec/data"


def dump_tfmap(qmap, fname, idt):
    with open(fname, 'w') as outfile:
        json.dump(qmap, indent=idt, separators=(',', ': '), fp=outfile)


def dump_dfmap(qmap, fname):
    dfmap = {term: len(qmap[term]) for term in qmap}
    dump_tfmap(dfmap, fname, 1)


def updatemap(qmap, words, docno):
    count_map = Counter(words)
    for key in count_map:
        update = {docno : count_map[key]}
        if key in qmap:
            qmap[key].update(update)
        else:
            qmap[key] = update



def writeStat(stats, out):
    wcount = stats[0]
    scount = stats[1]
    dcount = stats[2]

    fobj = open(out + "/" + "stats.txt", 'w')

    Docstr = "Documents: " + str(dcount)
    Sentstr = "Sentences: " + str(scount) + " (" + str(float(scount) / dcount) + "/doc)"
    Wordstr = "Words: " + str(wcount) + " (" + str(float(wcount) / scount) + "/sent)"

    # Write and Close the file
    fobj.write(Docstr + "\n")
    fobj.write(Sentstr + "\n")
    fobj.write(Wordstr + "\n")
    fobj.close()


def extract(fname, corpus, out):
    dcount = 0
    scount = 0
    wcount = 0
    qmap = dict()

    fobj = open(fname, 'w')

    for si in streamcorpus.Chunk(path=corpus):
        # print(si.body.clean_visible)
        docno = si.doc_id
        dcount += 1

        try:
            sentences = si.body.sentences["serif"]
        except KeyError:
            sentences = si.body.sentences["lingpipe"]
        scount += len(sentences)

        ## Get the tag-stripped text from 'clean_visible'
        if si.body.clean_visible is None or si.body.clean_html is None:
            continue

        ## Parse the html via boilerpipe
        doc_html = si.body.clean_html
        extractor = Extractor(extractor='ArticleExtractor', html=doc_html)
        text = extractor.getText()


        # Clean and update the term frequency map
        text = (onlyenglish(puncleft(puncright(text))))
        text = text.replace('\n', ' ').replace('\r', ' ').lower()

        words = text.split()
        wcount += len(words)

        updatemap(qmap, words, docno)
        doctext = " ".join(words)


        fobj.write(doctext)


    # Post processing
    tf_file = out + "/" + "TFmap.json"
    df_file = out + "/" + "DFmap.json"
    dump_tfmap(qmap, tf_file, 2)
    dump_dfmap(qmap, df_file)

    return (wcount, scount, dcount)


def read():
    for filename in glob(inpath + '/' + '*.xz'):
        spa = filename.split('/')
        dirname = spa[len(spa) - 1].split('.')[0]

        out = outpath + "/" + dirname
        if not os.path.isdir(out):
            os.makedirs(out)

        fileout = out + "/" + 'data.txt'
        print("Processing -> " + filename)
        stats = extract(fileout, filename, out)

        writeStat(stats, out)


if __name__ == '__main__':
    read()
