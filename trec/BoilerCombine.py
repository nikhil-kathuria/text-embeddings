import streamcorpus
from Cleanse import *
from BoilerExtract import updatemap, dump_dfmap, dump_tfmap, writeStat, clean
from boilerpipe.extract import Extractor
from glob import glob
import os


inpath="/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec"
outpath="/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec/data"


dcount = 0
scount = 0
wcount = 0


def extract(fobj, corpus, qmap):

    for si in streamcorpus.Chunk(path=corpus):
        global dcount
        global scount
        global wcount

        docno = si.doc_id
        dcount += 1


        ## Get the tag-stripped text from 'clean_visible'
        if si.body.clean_visible is None or si.body.clean_html is None:
            continue

        try:
            sentences = si.body.sentences["serif"]
        except KeyError:
            sentences = si.body.sentences["lingpipe"]
        scount += len(sentences)


        ## Parse the html via boilerpipe
        doc_html = si.body.clean_html
        extractor = Extractor(extractor='ArticleExtractor', html=doc_html)
        text = extractor.getText()


        # Clean and update the term frequency map
        text = clean(text)

        words = text.split()
        wcount += len(words)

        updatemap(qmap, words, docno)
        doctext = " ".join(words) + " "

        fobj.write(doctext)


def read():
    qmap = dict()

    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    fobj = open(outpath + "/" + "data.txt", 'w')

    for filename in glob(inpath + '/*'):

        print("Processing -> " + filename)
        extract(fobj, filename, qmap)

    tf_file = outpath + "/" + "TFmap.json"
    df_file = outpath + "/" + "DFmap.json"
    dump_tfmap(qmap, tf_file, 2)
    dump_dfmap(qmap, df_file)

    # Post processing
    fobj.close()
    writeStat((wcount, scount, dcount), outpath)



if __name__ == '__main__':
    read()