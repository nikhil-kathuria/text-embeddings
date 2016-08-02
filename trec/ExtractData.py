from glob import glob
import json
import os


import streamcorpus
from Cleanse import *

inpath="/Users/nikhilk/Documents/NEU_MSCS/MLLAB/Text_Vectors/trec/data"
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
        #print(si.body.clean_visible)
        doctext = ""
        docno = si.doc_id
        dcount += 1

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
            scount += 1
            sent = ""
            # print(" ".join([x.token for x in s.tokens]))
            for x in s.tokens:
                wcount +=1

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
                else:
                    updatemap(qmap, word, docno)



                # Form sentences and doc and finally write
                sent = sent + " " + word.strip()
            doctext = doctext + " " + sent.strip()
        fobj.write(shrinkspace(doctext) + " ")

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