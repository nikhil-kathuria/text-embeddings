
import streamcorpus
import json
from Cleanse import *

def performextract(name):
    fobj = open('whole.txt', 'w')
    holder = dict()
    num_docs = 0

    for si in streamcorpus.Chunk(path=name):
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
        obj = {'url': si.abs_url, 'body': si.body.clean_visible, 'sentences': sentencesa, 'source': si.source, 'stream_id': si.stream_id, 'stream_time': si.stream_time.zulu_timestamp, 'source_metadata': si.source_metadata, 'path': name}
        # obj['body'] =  shrinkspace(exceptall(removehtml(replacepath(replacemail(adjacentpuntatuions(onlyenglish(obj['body']))))))).lower()
        obj['body'] =  shrinkspace(exceptall(adjacentpuntatuions(emptypunctuations(replacepath(replacemail(removehtml(onlyenglish(obj['body'])))))))).lower()
        # obj['body'] =  shrinkspace(exceptwords(adjacentpuntatuions(emptypunctuations(replacepath(replacemail(removehtml(onlyenglish(obj['body'])))))))).lower()
        # print obj['body']
        holder[num_docs] = obj

    return holder



def mywrite(holder):
    whole = ""
    for obj in holder.values():
        line = re.sub('[\n\r]',' ',obj['body'])
        whole = whole + " " + line
    whole = shrinkspace(whole)
    fobj = open("trec.txt", 'w')
    fobj.write(whole)
    fobj.close()



if __name__ == '__main__':
    obj = performextract("./data/27.relonly.thrift.xz")
    mywrite(obj)
    # for key in obj.keys():
        # print str(key) + "->" +  str((obj[key]))
        # print  obj['sentences']