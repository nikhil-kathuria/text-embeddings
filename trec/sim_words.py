
import sys
import requests
import json


## Check http://thesaurus.altervista.org/ for more details on API

endpoint = "http://thesaurus.altervista.org/thesaurus/v1"
key = "f9kyG53iKSrvS1n9cT8J"
output = "json"
lang = "en_US"
termdict = json.load(open('DFmap.json'))



def apicall(word):
    wordset = set()
    word = word.strip()
    # .encode(encoding='utf-8')

    try:
        url = endpoint + "?word=" + word + "&language=" + lang + "&key=" + key + "&output="+output

        req = requests.get(url)
    except:
        print("Error Occuered in getting json response")
        return wordset

    if req.status_code == 200:
        res = req.json()
        for wlist in res['response']:
            try:
                words = wlist['list']['synonyms']
                words = words.split("|")
            except KeyError:
                continue
            wordset.update(words)

    return wordset


def multiple_words(wsplit, finalset, termset):
    newwords = ["-".join(wsplit), "_".join(wsplit), "".join(wsplit)]
    for word in newwords:
        if word in termset:
            finalset.add(word)
            ## Break if we thing a match justfies single entry
            #break


def findwords(term):
    termset = set(termdict.keys())
    if term not in termset:
        return

    finalset = set()
    wordset = apicall(term)
    print(wordset)

    for words in wordset:
        wsplit = words.split()
        if len(wsplit) > 1:
            multiple_words(wsplit, finalset, termset)
        else:
            if words in termset:
                finalset.add(words)

    ## Write to words.txt
    str =  term + " " +  " ".join(finalset)
    fobj = open('words.txt', 'a')
    fobj.write(str + "\n")
    fobj.close()




def checkargs():
    if len(sys.argv) == 2:
        findwords(sys.argv[1])
    else:
        print("Usage: python sim_words.py <WORD>")


if __name__ == '__main__':
    checkargs()
    #findwords('crash')
