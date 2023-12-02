from nltk import word_tokenize
from nltk.corpus import stopwords
from Tokenization import Tokenization
from part3 import *

if __name__ == '__main__':
    print("                  positional index               ")
    docslist = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt', '10.txt']
    b=Tokenization
    index_list={}
    index_list=b.get_p_index(docslist)
    for x, y in index_list.items():
        print(x, "=", y)
    print("                  	Term Frequency	               ")
    tfList = get_term_freq()
    print(tfList)

    print("                  	weighted Term Frequency	               ")
    wtfList = get_weighted_term_freq()
    print(wtfList)

    print("                  	inverse document Frequency	               ")
    idfList = get_inverse_document_freq()
    print(idfList)

    print("                  	      tf*idf	                           ")
    tfidfList = get_tf_idf()
    print(tfidfList)
    print("                                 doc length                      ")
    dl = get_doc_length()
    print(dl)
    print("                                 normalizid tf*idf                     ")
    n=get_normalized_tfidf()
    print(n)
    query = input("please enter the query: ")
    result = b.process_query(query, index_list)
    if len(result) > 0:
         x=1
    else:
         x=0
    if x==1:

        df1=get_result(query,result)
        print(df1[0])
        print(df1[1])
        print("Query Length = ",df1[2])
        sorted_s=sorted(df1[3],reverse=True)
        for i in range(len(df1[3])):
            print('cosine similarity(q,doc',df1[4][i],') =',df1[3][i])

        print("returned docs",sorted_s)
    else:
        print("try Another query")
