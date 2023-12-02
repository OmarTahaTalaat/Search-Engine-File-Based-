import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
class Tokenization:

    def __init__(self):
        self.query=[]

    def tokenization(self,query):
         stopWords = set(stopwords.words('english'))
         stopWords.remove('in')
         stopWords.remove('to')
         stopWords.remove('where')
         tokens = word_tokenize(query)
         filteredTokens = [word.lower() for word in tokens if word not in stopWords]
         return filteredTokens

    def tokenizationn(query):
         stopWords = set(stopwords.words('english'))
         stopWords.remove('in')
         stopWords.remove('to')
         stopWords.remove('where')
         tokens = word_tokenize(query)
         filteredTokens = [word.lower() for word in tokens if word not in stopWords]

         return filteredTokens

    def get_p_index(docslist) :
        index = {}

        for doc in docslist:
            f = open(doc)
            text = f.read()
            f.close()
            b=Tokenization()
            tokens=b.tokenization(text)
            for i in range(len(tokens)):
                if tokens[i] not in index.keys():
                    index[tokens[i]] = [[],{}]
                    index[tokens[i]][0] = 1
                    index[tokens[i]][1][doc] = []
                    index[tokens[i]][1][doc].append(i+1)
                else:

                    wordMap = index[tokens[i]][1]
                    if doc not in wordMap.keys():
                        index[tokens[i]][0] +=1
                         # Positions not added for that document
                        wordMap[doc] = []
                        wordMap[doc].append(i+1)
                        index[tokens[i]][1] = wordMap
        return index

    def retrieve_list(word,index):

        ans = []
        if word in  index.keys():

            ans =  index[word]
        else:
            print('Term : {} not present in dictionary'.format(word))
        return ans

    def check(res,post):
        listt=list(res[1].keys())
        keys=list(post[1].keys())
        s= [[],{}]
        for i in range(len(keys)):
            if(keys[i] in listt):

                for j in range(len(post[1][keys[i]])):
                    c=0
                    for k in range(len(res[1][keys[i]])):

                       if(post[1][keys[i]][j]==(res[1][keys[i]][k]+1)) :
                           a = post[1][keys[i]]
                           s[1][keys[i]] = a
                           c=1
                           break
                       else:
                           k+=1
                    if(c==1):
                        break
                    else:
                        j+=1

        res=s
        return s

    def process_query(query,index):
        b = Tokenization
        res= []
        test=[]
        query = query.lower()
        text = b.tokenizationn(query)
        if(len(text)==1):
           post= b.retrieve_list(text[0], index)
           if(post==[]):
               return res
           else:
               return list(post[1].keys())
        else:
            for i in range(len(text)):
                if(i==0):
                    post = b.retrieve_list(text[0], index)
                    if (post==[]):
                        return res
                    else:
                        res = post
                else:
                    post=b.retrieve_list(text[i],index)
                    if (post==[]):
                        return []
                    else:
                       test= b.check(res, post)
                    res=test
        return list(res[1].keys())
