import streamcorpus
from BoilerExtract import updatemap, dump_dfmap, dump_tfmap, writeStat, clean
from boilerpipe.extract import Extractor
from glob import glob
import os
from gensim.models import Phrases


inpath="/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec"
outpath="/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec/data"

# Variables for stats
dcount = 0
scount = 0
wcount = 0

# Gensim Bigram thresholds
bigram_mincount = 10
bigram_threshold = 5


def extract(dmap, pobj, fobj, corpus):
    global dcount
    global scount
    global wcount

    for si in streamcorpus.Chunk(path=corpus):
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

        # Write to doc tex map file
        fobj.write(corpus + "{|}" + docno + "{|}" + text)

        words = text.split()
        wcount += len(words)

        # Append final doc and update vocab for bigram filtering
        pobj.add_vocab([words])
        dmap[docno] = words


def read():
    qmap = dict()
    dmap = dict()
    pobj  = Phrases(min_count=bigram_mincount, threshold=bigram_threshold, delimiter='-')

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    fobj = open(outpath + "/" + "dmap.txt", 'w')

    # Iterate over all the files
    for filename in glob(inpath + '/*'):

        print("Processing -> " + filename)
        extract(dmap, pobj, fobj, filename)

    fobj.close()

    # Now write to file
    fobj = open(outpath + "/" + "data.txt", 'w')
    for key in dmap:
        words = pobj[dmap[key]]
        updatemap(qmap, words , key)
        doctext = " ".join(words) + " "
        fobj.write(doctext)
    fobj.close()


    ## Write TF and DF Map
    tf_file = outpath + "/" + "TFmap.json"
    df_file = outpath + "/" + "DFmap.json"
    dump_tfmap(qmap, tf_file, 2)
    dump_dfmap(qmap, df_file)

    ## Write Stats to file
    writeStat((wcount, scount, dcount), outpath)


if __name__ == '__main__':
    read()