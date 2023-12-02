import pandas as pd
import math
from Tokenization import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


document_length = pd.DataFrame()
docslist = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt', '10.txt']
tokens = []
all_words = []

for doc in docslist:
    f = open(doc)
    text = f.read()
    f.close()
    toknize = Tokenization()
    tokens.append(toknize.tokenization(text))

for doc in tokens:
    for word in doc:
        all_words.append(word)


def term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found


def get_term_freq():
    term_freq_list = pd.DataFrame(term_freq(tokens[0]).values(), index=term_freq(tokens[0]).keys())

    for i in range(1, len(tokens)):
        term_freq_list[i] = term_freq(tokens[i]).values()

    term_freq_list.columns = ['d' + str(i) for i in range(1, 11)]

    return term_freq_list.sort_index(ascending=True)


def weighted_term_freq(num):
    if num > 0:
        return math.log(num) + 1
    else:
        return 0


def get_weighted_term_freq():
    W_term_freq_list = get_term_freq()
    for i in range(1, len(tokens) + 1):
        W_term_freq_list['d' + str(i)] = W_term_freq_list['d' + str(i)].apply(weighted_term_freq)
    return W_term_freq_list.sort_index(ascending=True)


def get_inverse_document_freq():
    idf = pd.DataFrame(columns=['df', 'idf'])
    tf = get_term_freq()

    for i in range(len(tf)):
        freq = tf.iloc[i].values.sum()
        idf.loc[i, 'df'] = freq
        idf.loc[i, 'idf'] = math.log10(10 / freq)

    idf.index = tf.index
    return idf.sort_index(ascending=True)


def get_tf_idf():
    tf = get_term_freq()
    idf = get_inverse_document_freq()
    tfidf = tf.multiply(idf['idf'], axis=0)

    return tfidf.sort_index(ascending=True)


def doc_length(col):
    tfidf = get_tf_idf()
    return np.sqrt(tfidf[col].apply(lambda x: x ** 2).sum())


def get_doc_length():
    for column in get_tf_idf():
        document_length.loc[column + ' length', 'docs_length'] = doc_length(column)
    return document_length


def normalized_tfidf(col, x):
        return x / doc_length(col)


def get_normalized_tfidf():
    normalized_a = pd.DataFrame()
    tfidf = get_tf_idf()

    for column in tfidf.columns:
        normalized_a[column] = tfidf[column].apply(lambda x: normalized_tfidf(column, x))
    return normalized_a.sort_index(ascending=True)

def get_result(query,list1):
    b=Tokenization
    idfList=get_inverse_document_freq()
    no=get_normalized_tfidf()
    docs=[]
    for i in range(len(list1)):
        docs.append('d'+list1[i].replace('.txt',''))

    ind=b.tokenizationn(query)
    df2i=b.tokenizationn(query)
    df2i.append('sum')
    df1 = pd.DataFrame(columns=['tf','wt','idf','tf*idf','normalized'],index=ind)
    df2 = pd.DataFrame(columns=docs,index=df2i)
    for j in ind:
        df1.loc[j, 'tf'] = 1
        wt = 1+math.log10(1)
        df1.loc[j, 'wt'] = wt
        idf = idfList.loc[j, "idf"]
        df1.loc[j, 'idf'] = idf*wt
        tfidf=idf*wt
        df1.loc[j, 'tf*idf'] = wt*tfidf
    qLen=math.sqrt(sum(df1.loc[:,"idf"]**2))
    for i in range(len(docs)):
        sum1=0
        for j in ind:
            norm=no.loc[j,docs[i]]
            df1.loc[j, 'normalized'] = df1.loc[j,"idf"]/  math.sqrt(sum(df1.loc[:,"idf"]**2))
            df2.loc[j,docs[i]]=norm*df1.loc[j, 'normalized']
            sum1+=norm*df1.loc[j, 'normalized']
        df2.loc["sum", docs[i]] = sum1
        cs=[]
        for i in range(len(docs)):
            cs.append(df2.loc['sum',docs[i]])
    return df1,df2,qLen,cs,docs
