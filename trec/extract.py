
#!/usr/bin/python
"""
Reads Temporal Summarization Corpus Data
"""
## import standard libraries and parse command line args
import sys
import json
import time
#import urllib2
#import pprint

## import the thrifts library
import streamcorpus

## import the command line parsing library from python 2.7, can be
## installed on early python too.
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(dest="corpus", help="name of file to parse (used for corpus_id)")
parser.add_argument("--max", dest="max_docs", type=int, help="limit number of docs we examine")
parser.add_argument("-l", "--logfile", help="Log file")
args = parser.parse_args()


if args.logfile:
    logf = open(args.logfile, "a")
    #save_file = args.logfile.replace(".log",".resume")
    #if save_file == args.logfile:
    #    save_file = args.logfile + ".resume"
else:
    logf = sys.stderr


def log(mesg):
    #sys.stderr.write("%s\n" % mesg)
    #sys.stderr.flush()
    print >>logf, mesg

## set the corpus identifier in filter_run

## do the run
# keep track of elapsed time
start_time = time.time()
num_entity_doc_compares = 0
num_filter_results = 0
num_docs = 0
num_bytes = 0
num_stream_hours = 0


for si in streamcorpus.Chunk(path=args.corpus):
    ## count docs considered
    num_docs += 1

    ## get the tag-stripped text from 'clean_visible'
    if si.body.clean_visible is None:
        continue

    #pprint.pprint(si)
    #quit()

     # Here we concatenate the words into a space-separated string. Saves us the
     # hassle of cleaning and tokenizing the clean_visible text.
    try:
        sentences = si.body.sentences["serif"]
    except KeyError:
        sentences = si.body.sentences["lingpipe"]

     # One entry per sentence. Could concatenate further if desired.
    sentencesa = [" ".join([x.token for x in s.tokens]) for s in sentences]

     # If you want some metadata too
    if 'kba-2012' in si.source_metadata:
        try:
            si.source_metadata['kba-2012'] = json.loads(si.source_metadata['kba-2012'])
        except Exception:
            si.source_metadata['kba-2012'] = json.dumps(si.source_metadata['kba-2012'])

     # Relevant information.
    obj = {'url': si.abs_url, 'body': si.body.clean_visible, 'sentences': sentencesa, 'source': si.source, 'stream_id': si.stream_id, 'stream_time': si.stream_time.zulu_timestamp, 'source_metadata': si.source_metadata, 'path': args.corpus}


log("# done!")