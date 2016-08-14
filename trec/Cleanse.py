import re


def shrinkspace(text):
    return re.sub('\s+', ' ', text)


def adjacentpuntatuions(text):
    return re.sub('[;:`.!,"?_#$&%\'\-]{2,}', '', text)


def onlyenglish(text):
    return re.sub(r'[^\x00-\x7F]+',' ', text)


def replacedomain(text):
    #return re.sub('([^\.]+\.)+', '', text)
    return text


def replaceurl(text):
    return re.sub('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)


def replacemail(text):
    return re.sub('[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-zA-Z]{2,}', '', text)


def replacepath(text):
    return re.sub('(/[^/ ]*)+/?', '', text)

def exceptall(text):
    return re.sub(r'[^\w\s;:`.!,?_#$&\'\-]', '', text)


def removehtml(text):
    return re.sub('&amp;|&lt;|&gt;|&nbsp;|&iexcl;|&cent;|&pound;|&curren;|&yen;|&brvbar;|&sect;|&uml;|&copy;|&ordf|&laquo;|&not;|&shy;|&reg;|&macr;|&deg;|&plusmn;|&sup2|&sup3;|&acute;|&micro;|&para;|&middot;|&cedil;|&sup1;|&ordm;|&raquo;|&frac14;|&frac12;|&frac34;|&iquest;|&times;|&divide;|&ETH;|&eth;|&THORN;|&thorn;|&AElig;|&aelig;|&OElig;|&oelig;|&Aring;|&Oslash;|&Ccedil;|&ccedil;||&szlig;|&Ntilde;|&ntilde;','',text)
    # return re.sub('&amp|&gt' ,'', text)


def emptypunctuations(text):
    return re.sub('\s[;:`.!,"?_#$&%\'\-/]+\s', ' ', text)


def onlypunc(text):
    return re.sub('^[;:`.!,"?_#$&%\'\-/|<>(){}=@]+$', '', text)


def endpunc(text):
    return re.sub('[;:`.!,"?_#$&%\'\-/|<>(){}=@]+$', '', text)


def startpunc(text):
    return re.sub('^[;:`.!,"?_#$&%\'\-/|<>(){}=@]+', '', text)

def puncleft(text):
    return re.sub('\s+[;:`.!,"?_#$&%\'\-/|<>(){}=@]{1,}', ' ', text)

def puncright(text):
    return re.sub('[;:`.!,"?_#$&%\'\-/|<>(){}=@]{1,}\s+', ' ', text)



def exceptwords(text):
    return re.sub('[^\w]', ' ',text)


def shrinksapce(text):
    return re.sub('\s', ' ', text)


def remove_(text):
    return re.sub('[_]', ' ', text)


def all(text):
    return shrinkspace(adjacentpuntatuions(onlyenglish(replacedomain(replacedomain(replacepath(text))))))


if __name__ == '__main__':
    # print shrinkspace("as dasd asd a    asdas dasd as  d  \t asd  \n qwe")
    # print adjacentpuntatuions(";'; ;a a;' a' s! ';.my; U.S U.S.S.R -- --a")
    # print onlyenglish("Home \xc2\xbb South\xc2\xbb")
    # print replacemail("as d \t nik.hil@gmail.com")
    # print replaceurl("what http://www.facebook.com/share.php?u=<url> what the ")
    # print replacedomain('form Web www.colombopage.com ColomboPage')
    # print replacepath('my //var/www/ my')
    # print replacepath("/var/www/web_smashits/libs/db.php line Warning")
    # print replacepath('hello \var\dump\ whynot')
    # print exceptwords("news desk, sri lanka. oct 29, colombo: b")
    # print emptypunctuations("territories -- andaman ... a ")
    # print onlypunc(" -- ")
    # print endpunc("asd'';")
    # print startpunc("';asd'")
    # print puncleft(" :tom")
    # print puncright("mike. ")
    # print shrinkspace("  as     asd14 45554=-asd 12 4 5")
    #str='Sri Lanka evacuates coastline ahead of cyclone\nPosted: 30 October 2012 0356 hrs\n\xa0\n\xa0of\n\xa0 \xa0 \xa0\nCOLOMBO: Sri Lanka has ordered the evacuation of thousands of residents ahead a cyclone expected to hit the island\'s north-eastern coast early Tuesday, the Disaster Management Centre (DMC) said.\nPeople living within 500 metres (550 yards) from the coast were asked to move inland before the cyclone was due to make landfall at 2:00am Tuesday (2030 GMT), DMC director Sarath Lal Kumara said.\n"The evacuation applies to a coastline of about 150 kilometres (93 miles) in the island\'s north-east," he told AFP. "The area is thinly populated. We are issuing the evacuation notice as precaution to ensure there are no casualties."\nThe local meteorological department said a deep depression in the Bay of Bengal had developed into a "marginal cyclone" and was heading towards the island with wind speeds of up to 80 kilometres (50 miles) an hour.\nThe "marginal cyclone" was due to strike Sri Lanka\'s Mullaittivu district.\n"Low-lying coastal areas may be slightly inundated by sea waves," the meteorological department said. "Shallow and deep sea areas off the coast will experience very rough conditions, strong winds and intermittent rain."\nThe region is Sri Lanka\'s most sparsely populated area and was the scene of the final battle between government forces and Tamil Tiger rebels in May 2009. The area had been off limits for civilians till recently.\nOfficials said they used tsunami early warning towers in the area to alert people to move away from the coastline.\nHeavy winds and rain killed 19 people in the island\'s south in November last year.\nSri Lanka depends on monsoon rains for irrigation and power generation, but the seasonal downpours frequently cause death and property damage.\n- AFP/fa\n'
    str = "tragedy.,  tragedy,. "

    print(onlyenglish(puncleft(puncright(str))))
    #print(adjacentpuntatuions(str))