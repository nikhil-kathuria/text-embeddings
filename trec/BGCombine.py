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
bigram_mincount = 5
bigram_threshold = 2

# The list for all doc's text
dtext = []


def extract(pobj, corpus, qmap):
    global dcount
    global scount
    global wcount
    global dtext

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

        words = text.split()
        wcount += len(words)

        updatemap(qmap, words, docno)

        # Append final doc and update vocab for bigram filtering
        pobj.add_vocab([words])
        dtext.append(words)




def read():
    qmap = dict()
    pobj  = Phrases(min_count=bigram_mincount, threshold=bigram_threshold, delimiter='-')

    if not os.path.isdir(outpath):
        os.makedirs(outpath)


    for filename in glob(inpath + '/*'):

        print("Processing -> " + filename)
        extract(pobj, filename, qmap)

    tf_file = outpath + "/" + "TFmap.json"
    df_file = outpath + "/" + "DFmap.json"
    dump_tfmap(qmap, tf_file, 2)
    dump_dfmap(qmap, df_file)

    # Now write to file via
    fobj = open(outpath + "/" + "data.txt", 'w')
    for line in dtext:
        dlist = pobj[line]
        doctext = " ".join(dlist) + " "
        fobj.write(doctext)
    fobj.close()
    writeStat((wcount, scount, dcount), outpath)



if __name__ == '__main__':
    read()