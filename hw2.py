# convert rare words 
import json
from collections import defaultdict
from count_cfg_freq import Counts

class pCounters(Counts):
    def __init__(self):
        super(pCounters,self).__init__()
        self.words = defaultdict(int)
        self.qbinary = defaultdict(float)
        self.qunary = defaultdict(float)
        self.symbinary = defaultdict(list)

    def readin(self,dat):
        if len(dat) not in set([3,4,5]):
            print "readin line has wrong content! ",dat
        elif dat[1] == 'NONTERMINAL':
            self.nonterm[dat[2]] = int(dat[0])
        elif dat[1] == 'UNARYRULE':
            self.unary[(dat[2],dat[3])] = int(dat[0])
        elif dat[1] == 'BINARYRULE':
            self.binary[(dat[2],dat[3],dat[4])] = int(dat[0])
        else:
            print "something wrong! ",dat

    def countWord(self):
        for (sym,word), count in self.unary.iteritems():
            self.words[word] += count

    def countQ(self):
        for (sym, word), count in self.unary.iteritems():
            self.qunary[(sym,word)] = float(count)/self.nonterm[sym]
        for (sym, y1, y2), count in self.binary.iteritems():
            self.qbinary[(sym,y1,y2)] = float(count)/self.nonterm[sym]
            self.symbinary[sym].append((y1,y2))
    
def wordToRare(tree,count):
    if isinstance(tree,basestring): return
    if len(tree) == 3:
        wordToRare(tree[1],count)
        wordToRare(tree[2],count)
    elif len(tree) == 2:
        if count.words[tree[1]] < 5:
            tree[1] = '_RARE_'

def convertRare(filename):
    pc = pCounters()
    for l in open(filename+'.counts.dat','r'):
        pc.readin(l.split())
    pc.countWord()
    with open(filename+'_r.dat','w') as f:
        for l in open(filename+'.dat','r'):
            t = json.loads(l)
            wordToRare(t,pc)
            f.write(json.dumps(t)+'\n')

def CKY(sentence,count):
    '''
    CKY parsing algorithm

    :param sentence: sentence list
    :param count: training model
    :returns json structure of the data
    '''
    # initialize
    pi = defaultdict(float)
    bp = defaultdict(int)
    n = len(sentence)
    for i,word in enumerate(sentence):
        if word in count.words: tword = word
        else: tword = '_RARE_'
        for symbol in count.nonterm:
            pi[(i,i,symbol)] = count.qunary[(symbol,tword)]
    # loop increasing length
    coef = 1.
    for l in range(1,n-1):
        for i in range(n-l):
            j = i+l
            for symbol in count.nonterm:
                pimax = 0.0
                for rule in count.symbinary[symbol]:
                    for s in range(i,j):
                        tpi = count.qbinary[(symbol,rule[0],rule[1])]*pi[(i,s,rule[0])]*pi[(s+1,j,rule[1])]
                        if tpi > pimax:
                            bp[(i,j,symbol)] = (rule[0],rule[1],s)
                            pi[(i,j,symbol)] = tpi*coef
                            pimax = tpi
    # Last loop, search for 'SBARQ'
    i,j = 0, n-1
    symbol='SBARQ'
    pimax = 0.
    for rule in count.symbinary[symbol]:
        for s in range(i,j):
            tpi = count.qbinary[(symbol,rule[0],rule[1])]*pi[(i,s,rule[0])]*pi[(s+1,j,rule[1])]
            if tpi > pimax:
                bp[(i,j,symbol)] = (rule[0],rule[1],s)
                pi[(i,j,symbol)] = tpi*coef
                pimax = tpi
    
    print pi[(i,j,symbol)]
    return restructTree(bp,i,j,symbol,sentence)

def restructTree(bp,i,j,symbol,sentence):
    ''' reconstruct the Tree structure of sentence

    :param bp: backpointers from CKY
    :param i: start of mark
    :param j: end of mark
    :param sentence: input sentence
    '''
    if i > j:
        print "!indexing wrong! @ restructTree"
        exit
    if i == j:
        return [symbol,sentence[i]]
    else:
        r1,r2,s = bp[(i,j,symbol)]
        return [symbol,restructTree(bp,i,s,r1,sentence),restructTree(bp,s+1,j,r2,sentence)]

if __name__ == "__main__":
    #convertRare('parse_train')

    # prepare model
    pc = pCounters()
    for l in open('parse_train_r.counts.dat','r'):
        pc.readin(l.split())
    pc.countWord()
    pc.countQ()

    # analyze sentence
    with open('parse_dev_output.dat','w') as f:
        for l in open('parse_test.dat','r'):
            f.write(json.dumps(CKY(l.split(),pc))+'\n')


