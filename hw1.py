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
                tw = ''
                for w in fullset:
                    tmp = pi[(k-1,w,u)]*model.ngram2[(w,u,v)]*model.emission[(str[k-1],v)]
                    if tmp > pimax:
                        pimax = tmp
                        tw = w
                pi[(k,u,v)] = pimax
                bp[(k,u,v)] = tw
    pimax = 0
    for u in fullset:
        for v in fullset:
            tmp = pi[(n,u,v)]*model.ngram2[(u,v,'STOP')]
            if tmp > pimax:
                tu = u
                tv = v
                pimax = tmp

    # back construction
    state.append(tv)
    state.append(tu)
    for k in range(n-2,0,-1):
        state.append(bp[(k+2,state[-1],state[-2])])
        
    print 'max likelihood:',pimax        
    print str
    print state[::-1]
            

def qz6():
    # Initialize
    model = hmm.Hmm(3)
    input = file("qz6.counts",'r')
    model.read_counts(input)
    model.processing()
    # solve the problem
    str = 'the cat saw the saw'.split(" ")
    viterbi(str,model)

def qz7():
    
    
if __name__ == "__main__":
    qz7()
    
