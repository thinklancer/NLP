'''

'''
import sys
from collections import defaultdict
from IPython import embed

def printsentence(sentence):
    '''  print sentence
    '''
    sen = ' '.join(sentence)
    print sen.decode('utf-8')


def loadcorpus(filename):
    ''' load corpus file

    :param filename: filename to read
    :returns: data,n
    
         * data: sentence array
         * n : total number of sentence
    '''
    k = 0
    data = {}
    with open(filename) as f:
        for line in f:
            data[k] = line.split(' ')
            data[k][-1] = data[k][-1][:-1] # remove last '\n'
            k +=1
    return data, k
    
def addNULL(data,n):
    ''' extend the English corpus by NULL
    '''
    for i in range(n):
        data[i].extend(['NULL'])


def buildLib(corpus):
    ''' convert words from sentence to dictionary

    :param corpus: corpus from loadcorpus
    :returns : set of words
    '''

    dic = set()
    for line in corpus:
        for word in corpus[line]:
            dic.add(word)
    return dic

def initialTforIBM1(lang1,lang2,n):
    '''
    initialize the transfer table for IBM 1 model

    :param lang1: language 1
    :param lang2: language 2
    :param n: number of sentence
    '''
    ncount = defaultdict(int)
    for k in range(n):
        for word in lang1[k]:
            ncount[word] += len(lang2[k])

    t = defaultdict(float)
    c2 = defaultdict(float)
    c1 = defaultdict(float)
        
    for k in range(n):
        tempsum = 0.0
        for word1 in lang1[k]:
            tempsum += ncount[word1]
        for word2 in lang2[k]:
            for word1 in lang1[k]:
                # loop over for assigning c corpus
                c2[(word1,word2)] += ncount[word1]/tempsum
                c1[word1] += ncount[word1]/tempsum
    # assignment of t
    # have duplicates but quick than compare full word list
    for k in range(n):
        for word2 in lang2[k]:
            for word1 in lang1[k]:    
                t[(word2,word1)] = c2[(word1,word2)]/c1[word1]
    return t

def initialTforIBM2(lang1,lang2,n):
    '''
    initialize the transfer table for IBM 1 model

    :param lang1: language 1
    :param lang2: language 2
    :param n: number of sentence
    '''
    ncount = defaultdict(int)
    for k in range(n):
        for word in lang1[k]:
            ncount[word] += len(lang2[k])

    t = defaultdict(float)
    q = defaultdict(float)
    c2 = defaultdict(float)
    c1 = defaultdict(float)
        
    for k in range(n):
        tempsum = 0.0
        for word1 in lang1[k]:
            tempsum += ncount[word1]
        for j,word2 in enumerate(lang2[k]):
            for i,word1 in enumerate(lang1[k]):
                # loop over for assigning c corpus
                c2[(word1,word2)] += ncount[word1]/tempsum
                c1[word1] += ncount[word1]/tempsum
                q[(j,i,len(lang1[k]),len(lang2[k]))]= c2[(word1,word2)]/c1[word1]
    # assignment of t
    # have duplicates but quick than compare full word list
    for k in range(n):
        for word2 in lang2[k]:
            for word1 in lang1[k]:    
                t[(word2,word1)] = c2[(word1,word2)]/c1[word1]
    return t,q

def updateTforIBM1(lang1,lang2,n,t):
    '''
    update the transfer table for IBM 1 model

    :param lang1: language 1
    :param lang2: language 2
    :param n: number of sentence
    :param t: the previous transfer table
    '''
    c2 = defaultdict(float)
    c1 = defaultdict(float)
        
    for k in range(n):
        for word2 in lang2[k]:
            tempsum = 0.0
            for word1 in lang1[k]:
                tempsum += t[(word2,word1)]
            for word1 in lang1[k]:
                # loop over for assigning c corpus
                c2[(word1,word2)] += t[(word2,word1)]/tempsum
                c1[word1] += t[(word2,word1)]/tempsum
    for k in range(n):
        for word2 in lang2[k]:
            for word1 in lang1[k]:    
                t[(word2,word1)] = c2[(word1,word2)]/c1[word1]
    return t

def updateTforIBM2(lang1,lang2,n,t,q):
    '''
    update the transfer table for IBM 1 model

    :param lang1: language 1
    :param lang2: language 2
    :param n: number of sentence
    :param t: the previous transfer table
    '''
    c2 = defaultdict(float)
    c1 = defaultdict(float)
        
    for k in range(n):
        for j,word2 in emumerate(lang2[k]):
            tempsum = 0.0
            for i,word1 in enumerate(lang1[k]):
                tempsum += t[(word2,word1)]*q[(j,i,len(lang1[k]),len(lang2[k]))]
            for i,word1 in enumerate(lang1[k]):
                # loop over for assigning c corpus
                c2[(word1,word2)] += t[(word2,word1)]*q[(j,i,len(lang1[k]),len(lang2[k]))]/tempsum
                c1[word1] += t[(word2,word1)]*q[(j,i,len(lang1[k]),len(lang2[k]))]/tempsum
    for k in range(n):
        for word2 in lang2[k]:
            for word1 in lang1[k]:    
                t[(word2,word1)] = c2[(word1,word2)]/c1[word1]
    return t

def findAlignment(t,lang1,lang2,n,filename):
    ''' interpret the translation and assign translation pairs

    :param t: the previous transfer table
    :param lang1: language 1
    :param lang2: language 2
    :param n: number of sentence
    :param filename: output filename
    '''
    with open(filename,'w') as f:
        for k in range(n):
            for i,word2 in enumerate(lang2[k]): # loop forign word
                maxt = -1
                for j,word1 in enumerate(lang1[k]):
                    if t[(word2,word1)] > maxt:
                        maxt = t[(word2,word1)]
                        if word1 == 'NULL':
                            ai = -1
                        else:
                            ai = j
                if maxt != -1 and ai != -1:
                    f.write("{0} {1} {2}\n".format(k+1,ai+1,i+1))

def ibm1(lang1,lang2,n):
    ''' train data by IBM 1 model

    :param lang1: language 1
    :param lang2: language 2
    :param n: number of sentence
    '''
    t = initialTforIBM1(lang1,lang2,n)
    print "finish one loop"
    updateTforIBM1(lang1,lang2,n,t)
    print "finish one loop"
    updateTforIBM1(lang1,lang2,n,t)
    print "finish one loop"
    updateTforIBM1(lang1,lang2,n,t)
    print "finish one loop"
    updateTforIBM1(lang1,lang2,n,t)
    print "finish one loop"
    return t

def ibm2(lang1,lang2,n,nloop=5):
    ''' train data by IBM 1 model

    :param lang1: language 1
    :param lang2: language 2
    :param n: number of sentence
    '''
    t = initialTforIBM2(lang1,lang2,n)
    print "finish initialization"
    for i in range(nloop-1):
        updateTforIBM1(lang1,lang2,n,t)
        print "finish one loop"
    return t

def assignment2():
    filename = 'corpus'
    eng , n = loadcorpus(filename+'.en')
    esp , n2 = loadcorpus(filename+'.es')

    if n != n2:
        print "The file is correpted!!"
        exit

    addNULL(eng,n)
    t=ibm2(eng,esp,n)

    filename = 'test'
    lang1, n =  loadcorpus(filename+'.en')
    lang2, n =  loadcorpus(filename+'.es')
    findAlignment(t,lang1,lang2,n,'test.key')
    return 0

def assignment1():
    filename = 'corpus'
    eng , n = loadcorpus(filename+'.en')
    esp , n2 = loadcorpus(filename+'.es')

    if n != n2:
        print "The file is correpted!!"
        exit

    addNULL(eng,n)
    t=ibm1(eng,esp,n)

    filename = 'test'
    lang1, n =  loadcorpus(filename+'.en')
    lang2, n =  loadcorpus(filename+'.es')
    addNULL(lang1,n)
    findAlignment(t,lang1,lang2,n,'test2.key')
    return 0

if __name__ ==  "__main__":
    assignment1()
