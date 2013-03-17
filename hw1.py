''' solution file for homework 1
'''

import sys
from collections import defaultdict
import hmm

def qz6():
    # Initialize
    model = hmm.Hmm(3)
    with open('qz6.txt','r') as f:
        model.train(f)
    with open('qz6.counts.txt','w') as f:
        model.write_counts(f)
    model.read_counts_from_file("qz6.counts.txt")
    model.processing()
    # solve the problem
    str = 'the cat saw the saw'.split(" ")
    hmm.viterbi(str,model)  

def unitagger(model,target_file):
    ''' Unigram tagger program

    :param train_file: training set processed by :func:`hmm.Hmm.write_counts`
    :param target_file: file to tag
    '''
    alltag = model.all_states
    file = target_file+'.p1.out'
    with open(target_file,'r') as f:
        g = open(file,'w')
        for l in f:
            line = l.strip()
            if not line:
                g.write("\n")
            else:
                w = line
                #tag = 'I-GENE'
                maxt = 0
                for tmp in alltag:
                    if not w in model.words:
                        tw = '_RARE_'
                    else:
                        tw = w
                    if model.emission[tw,tmp] > maxt:
                        maxt = model.emission[tw,tmp]
                        tag = tmp
                g.write(w+' '+tag+'\n')
        g.close()

def viterbiTagger(model,target_file,file):
    ''' Viterbi tagger program

    :param train_file: training set processed by :func:`hmm.Hmm.write_counts`
    :param target_file: file to tag
    '''
    with open(target_file) as f:
        sentence_iterator = hmm.sentence_iterator(hmm.simple_conll_corpus_iterator(f))
        g = open(file,'w')
        for sentence in sentence_iterator:
            sent = ['*']+[word[1] for word in sentence]+['STOP']   # 1 is specified for this sentence iterator
            #state = hmm.viterbiClass(sent,model)
            state = hmm.viterbi(sent,model)
            #print ' '.join(sent)
            #print ' '.join(state)
            #print '************************'
            n = len(sent)
            for i in range(1,n-1):
                g.write(sent[i]+' '+state[i]+'\n')
            g.write('\n')
        g.close()
            

def hw1():
    model = hmm.Hmm(3)
    model.read_counts_from_file('gene.rare.counts') # python count_freqs.py gene.rare > gene.rare.counts
    model.processing()
    unitagger(model,'gene.dev')


def hw2():
    hmm.tagRare('gene.train') # --> gene.rare
    model = hmm.Hmm(3)
    with open('gene.rare','r') as f:
        model.train(f)
    with open('gene.rare.counts','w') as f:
        model.write_counts(f)
    model.read_counts_from_file('gene.rare.counts') # python count_freqs.py gene.rare > gene.rare.counts
    model.processing()
    viterbiTagger(model,'gene.dev','gene_dev.p2.out')

def hw3():
    hmm.tagClass('gene.train') # --> gene.rare2
    
    model = hmm.Hmm(3)
    with open('gene.rare2','r') as f:
        model.train(f)
    with open('gene.rare2.counts','w') as f:
        model.write_counts(f)
    model.read_counts_from_file('gene.rare2.counts') # python count_freqs.py gene.rare > gene.rare.counts
    model.processing()
    viterbiTagger(model,'gene.test','gene_test.p3.out')
    
if __name__ == "__main__":
    #qz6()
    hw2()
    #hw3()

