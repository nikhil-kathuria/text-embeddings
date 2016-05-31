import re


def shrinkspace(text):
    return re.sub('\s+', ' ', text)


def adjacentpuntatuions(text):
    return re.sub('[;:`.!,"?_#$&%\'\-]{2,}', '', text)


def onlyenglish(text):
    return re.sub(r'[^\x00-\x7F]+','', text)


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
    return re.sub('\s[;:`.!,"?_#$&%\'\-]+\s', ' ', text)


def exceptwords(text):
    return re.sub('[^\w]', ' ',text)

def shrinksapce(text):
    return re.sub('\s', ' ', text)

def remove_(text):
    return re.sub('[_]', ' ', text)



def all(text):
    return shrinkspace(adjacentpuntatuions(onlyenglish(replacedomain(replacedomain(replacepath(text))))))


if __name__ == '__main__':
    print shrinkspace("as dasd asd a    asdas dasd as  d  \t asd  \n qwe")
    print adjacentpuntatuions(";'; ;a a;' a' s! ';.my; U.S U.S.S.R -- --a")
    print onlyenglish("Home \xc2\xbb South\xc2\xbb")
    print replacemail("as d \t nik.hil@gmail.com")
    print replaceurl("what http://www.facebook.com/share.php?u=<url> what the ")
    print replacedomain('form Web www.colombopage.com ColomboPage')
    print replacepath('my //var/www/ my')
    print replacepath("/var/www/web_smashits/libs/db.php line Warning")
    print replacepath('hello \var\dump\ whynot')
    print exceptwords("news desk, sri lanka. oct 29, colombo: b")
    print emptypunctuations("territories - andaman ")
    print shrinkspace("  as     asd14 45554=-asd 12 4 5")