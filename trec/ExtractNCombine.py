import streamcorpus
from Cleanse import *
from ExtractData import updatemap, dump_dfmap, dump_tfmap
from glob import glob
import os


inpath="/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec"
outpath="/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec/data"


def extract(fobj, corpus, qmap):
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
                word = onlyenglish(replacemail(x.token)).lower()
                word = onlypunc(endpunc(word)).strip()
                # word = replaceurl(replacemail(x.token))
                # word = remove_(exceptwords(word)).lower()

                if word == "" or re.match('\s+', word):
                    continue

                spl = word.split('\s+')
                if len(spl) > 1:
                    for itr in range(len(spl)):
                        updatemap(qmap, spl[itr], docno)
                    word = ' '.join(spl)
                    print(word)

                else:
                    updatemap(qmap, word, docno)

                # Form sentences and doc and finally write
                sent = sent + " " + word.strip()
            doctext = doctext + " " + sent.strip()
        fobj.write(shrinkspace(doctext) + " ")




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



if __name__ == '__main__':
    read()