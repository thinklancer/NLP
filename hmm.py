#! /opt/local/bin/python
'''
Based on the homework code by Daniel Bauer <bauer@cs.columbia.edu>
'''

import sys
from collections import defaultdict
import math
import re

"""
Count n-gram frequencies in a data file and write counts to stdout.
=====================================================================
"""

def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else: # Empty line
            yield (None, None)                        
        l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in xrange(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        

'''
Class for HMM
=============
'''

class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()
        self.emission = defaultdict(int)
        self.ngram2 = defaultdict(int)
        self.words = set()
        
    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags            
            for i in xrange(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:            
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))


        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))

    def read_counts(self, corpusfile):
        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()
        self.words = set()

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
                self.words.add(word)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count

    def processing(self):
        ''' calculate emission etc based on counters
        '''
        for w in self.emission_counts:
            self.emission[w] = self.emission_counts[w]/self.ngram_counts[0][(w[1],)]
        # calculate q(s|u,v) = c(u,v,s)/c(u,v)
        for w in self.ngram_counts[2]:
            self.ngram2[w] = self.ngram_counts[2][w]/self.ngram_counts[1][w[:2]]

    def read_counts_from_file(self,file):
        ''' read counts from file name instead of file pointer
        '''
        with open(file,'r') as f:
            self.read_counts(f)

'''
Viterbi Algorithm
==================
'''
def simpleRare(w):
    '''
    return _RARE_ tag for any words input
    '''
    return '_RARE_'

def rare4Class(w):
    ''' return the rare class of given word w
    '''
    if bool(re.search('[0-9]',w)):
        return '_Numeric_'
    if bool(re.match("^[A-Z]*$",w)):
        return '_AllCaptials_'
    if bool(re.match("^[a-zA-Z]*$",w)) and w[-1].upper() == w[-1]:
        return '_LastCapital_'
    return '_RARE_'

def tagRareClass(corpus_file,rareClass):
    ''' tag rare word with count < 5
        output to the *file*.rare

    :param corpus_file: input file
    :param rareClass: the choice of Rare class
    '''
    file = corpus_file.split('.')[0]+'.rare'
    counts = defaultdict(int)
    with open(corpus_file,'r') as f:
        line_iterator = simple_conll_corpus_iterator(f)
        for l in line_iterator:
            counts[l[0]] += 1

    with open(corpus_file,'r') as f:
        line_iterator = simple_conll_corpus_iterator(f)
        g = open(file,'w')
        for w,t in line_iterator:
            if (w,t) == (None,None):
                g.write("\n")
            else:
                if counts[w] < 5:
                    g.write(rareClass(w)+' '+t+'\n')
                else:
                    g.write(w+' '+t+'\n')
        g.close()


def viterbi(str,model,rareClass):
    '''
    .. function:: viterbi(str, model [, rareClass=simpleRare])

    Viterbi Algorithm with Backpointer

    :param str: sentence for phrase
    :param model: language model from training set
    :param rareClass: the choice of Rare class
    :returns: states of the sentence
    '''
    
    # initialize
    boost = 4. # boost factor to avoid precision with tiny likelihood
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
                    if str[k-1] in model.words:
                        tmp = boost*pi[(k-1,w,u)]*model.ngram2[(w,u,v)]*model.emission[(str[k-1],v)]
                    else:
                        tmp = boost*pi[(k-1,w,u)]*model.ngram2[(w,u,v)]*model.emission[(rareClass(str[k-1]),v)]
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
        
    #print 'max likelihood:',pimax/boost**n       
    #print str
    #print state[::-1]
    return state[::-1]

def viterbiTagger(model,target_file,file,rareClass):
    ''' Viterbi tagger program

    :param train_file: training set processed by :func:`hmm.Hmm.write_counts`
    :param target_file: file to tag
    :param file: file to write the word-state pairs
    :param rareClass: the choice of Rare class
    '''
    with open(target_file) as f:
        sentence_iter = sentence_iterator(simple_conll_corpus_iterator(f))
        g = open(file,'w')
        for sentence in sentence_iter:
            sent = ['*']+[word[1] for word in sentence]   # 1 is specified for this sentence iterator
            state = viterbi(sent,model,rareClass)
            #print ' '.join(sent)
            #print ' '.join(state)
            #print '************************'
            n = len(sent)
            for i in range(1,n):
                g.write(sent[i]+' '+state[i]+'\n')
            g.write('\n')
        g.close()

if __name__ == "__main__":
    print "HMM model file"
