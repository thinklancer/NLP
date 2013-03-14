''' solution file for homework 1
'''

import sys
from collections import defaultdict
import hmm

def viterbi(str,model):
    ''' Viterbi Algorithm with Backpointer
    '''
    # initialize
    n = len(str)
    pi = defaultdict(int)
    bp = defaultdict(int)
    pi[(0,'*','*')] = 1
    state = []
    # Algorithm
    fullset = model.all_states | set('*')
    for k in range(1,n+1):
        for u in fullset:
            for v in fullset:
                pimax = 0
                for w in fullset:
                    tmp = pi[(k-1,w,u)]*model.ngram2[(w,u,v)]*model.emission[(str[k-1],v)]
                    #print k,w,u,v,tmp,str[k-1],model.ngram2[(w,u,v)],model.emission[(str[k-1],v)]
                    if tmp >= pimax:
                        pimax = tmp
                        tw = w
                pi[(k+1,u,v)] = pimax
                bp[(k+1,u,v)] = w
    # back construction
    pimax = 0
    for u in fullset:
        for v in fullset:
            tmp = pi[(n+1,u,v)]*model.ngram2[(u,v,'STOP')]
            print u,v,pi[(n+1,u,v)],model.ngram2[(u,v,'STOP')]
            if tmp > pimax:
                tu = u
                tv = v
                pimax = tmp
    print tu,tv,pimax
            

def qz6():
    # Initialize
    model = hmm.Hmm(3)
    input = file("qz6.counts",'r')
    model.read_counts(input)
    model.processing()
    str = 'the cat saw the saw'.split(" ")
    print model.ngram2
    viterbi(str,model)
    
if __name__ == "__main__":
    qz6()
    
