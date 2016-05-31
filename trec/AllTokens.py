
import streamcorpus
from Cleanse import *

def runExtract(fname, corpus):
    fobj = open(fname, 'w')
    count=0

    for si in streamcorpus.Chunk(path=corpus):
        count = count + 1
        print(si.doc_id)


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
                sent = sent + " " + word

            # fobj.write(shrinkspace(sent).strip() + " ")

        '''
         # If you want some metadata too
        if 'kba-2012' in si.source_metadata:
            try:
                si.source_metadata['kba-2012'] = json.loads(si.source_metadata['kba-2012'])
            except Exception:
                si.source_metadata['kba-2012'] = json.dumps(si.source_metadata['kba-2012'])
         # Relevant information.
        obj = {'url': si.abs_url, 'body': si.body.clean_visible, 'sentences': sentencesa, 'source': si.source, 'stream_id': si.stream_id, 'stream_time': si.stream_time.zulu_timestamp, 'source_metadata': si.source_metadata, 'path': args.corpus}
        '''

if __name__ == '__main__':
    runExtract('whole.txt', "27.relonly.thrift.xz")