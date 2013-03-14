''' solution file for homework 1
'''

import sys
from collections import defaultdict
import Hmm

def viterbi(str,model):
    ''' Viterbi Algorithm with Backpointer
    '''
    # initialize
    n = len(str)
    pi = defaultdict(int)
    pi[(0,'*','*')] = 1
    # Algorithm
    # k = 1 case
    u = '*'
    for v in model.all_states:
        pi[(1,u,v)] = model.ngram2[('*','*',v)]*model.emission[(str[0],v)]
    print pi
    # k > 1 cases

def qz6():
    # Initialize
    model = Hmm.Hmm(3)
    input = file("qz6.counts",'r')
    model.read_counts(input)
    model.processing()
    str = 'the cat saw the saw'.split(" ")
    viterbi(str,model)
    
if __name__ == "__main__":
    qz6()
    
