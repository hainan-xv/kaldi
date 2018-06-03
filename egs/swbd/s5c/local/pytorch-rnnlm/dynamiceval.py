import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
#import data

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')

parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str,
                    help='name of model to eval')
parser.add_argument('--gpu', type=int, default=0,
                    help='set gpu device ID (-1 for cpu)')
parser.add_argument('--val', action='store_true',
                    help='set for validation error, test by default')
parser.add_argument('--lamb', type=float, default=0.002,
                    help='decay parameter lambda')
parser.add_argument('--epsilon', type=float, default=0.00002,
                    help='stabilization parameter epsilon')
parser.add_argument('--lr', type=float, default=0.00005,
                    help='learning rate eta')
parser.add_argument('--oldhyper', action='store_true',
                    help='Transforms hyperparameters, equivalent to running old version of code')
parser.add_argument('--grid', action='store_true',
                    help='grid search for best hyperparams')
parser.add_argument('--gridfast', action='store_true',
                    help='grid search with partial validation set')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size for gradient statistics')
parser.add_argument('--bptt', type=int, default=5,
                    help='sequence/truncation length')
parser.add_argument('--max_batches', type=int, default=-1,
                    help='maximum number of training batches for gradient statistics')
parser.add_argument('--QRNN', action='store_true',
                    help='Use if model is a QRNN')
parser.add_argument('--test_data', type=str, default='dev/null',
                    help='test file')

args = parser.parse_args()

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

dictionary = Dictionary()

class Corpus(object):
    def __init__(self, path):
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.train_small = self.tokenize(os.path.join(path, 'train_small.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = dictionary.word2idx[word]
                    token += 1

        return ids

if args.gpu>=0:
    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
else:
    #to run on cpu, model must have been trained on cpu
    args.cuda=False

model_name=args.model

print('# loading')

corpus = Corpus(args.data)
eval_batch_size = 1

def read(fname, unk_index):                                                     
  """                                                                           
  Read a file where each line is of the form "word1 word2 ..."                  
  Yields lists of the form [word1, word2, ...]                                  
  """                                                                           
  w2i = dictionary.word2idx
  with open(fname, "r") as fh:                                                  
    for line in fh:                                                             
      sent = [w2i.get(x, unk_index) for x in line.strip().split()]              
      sent.insert(0, w2i["<eos>"])
      sent.append(w2i["<eos>"])                                                   
      yield torch.LongTensor(sent)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data
#######################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def get_batch_sent(source, i, evaluation=False):
#    seq_len = min(args.bptt, len(source) - 1 - i)
    seq_len = source[i].size(0)
    data = Variable(source[i][0:seq_len - 1], volatile=evaluation)
    target = Variable(source[i][1:seq_len].view(-1))
    return data, target

def evaluate():
    if args.QRNN:
       model.reset()

    total_loss = 0

    ntokens = len(dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    last = False
    seq_len= args.bptt
    seq_len0 = seq_len
    #loops through data

    batch_size = 128
    while i < len(eval_data):
        model.eval()
        this_size = min(batch_size, len(eval_data) - i)

        data_vec = []
        targets_vec = []
        length_vec = []
        for i in range(0, this_size):
          data, targets = get_batch_sent(eval_data,i)
          data_vec.append(data)
          targets_vec.append(targets)

          seq_len = targets.size(0)
          length_vec.append(seq_len)

        hidden = repackage_hidden(hidden)

        model.zero_grad()

        #assumes model has atleast 2 returns, and first is output and second is hidden
        returns = model(data, hidden)
        output = returns[0]
        hidden = returns[1]
        loss = criterion(output.view(-1, ntokens), targets)

        #compute gradient on sequence segment loss
#        loss.backward()
#
#        #update rule
#        for param in model.parameters():
#            dW = lamb*param.decrate*(param.data0-param.data)-lr*param.grad.data/(param.MS+epsilon)
#            param.data+=dW

        #seq_len/seq_len0 will be 1 except for last sequence
        #for last sequence, we downweight if sequence is shorter
        total_loss += loss.data
        batch += seq_len
        i += 1
        print ("sentence cost:", float(loss.data))

    #since entropy of first token was never measured
    #can conservatively measure with uniform distribution
    #makes very little difference, usually < 0.01 perplexity point
    #total_loss += (1/seq_len0)*torch.log(torch.from_numpy(np.array([ntokens])).type(torch.cuda.FloatTensor))
    #batch+=(1/seq_len0)

    perp = torch.exp(total_loss/batch)
    if args.cuda:
        return perp.cpu().numpy()
    else:
        return perp.numpy()

#load model
with open(model_name, 'rb') as f:
#    model = torch.load(f)
    model = torch.load(f, map_location=lambda storage, loc: storage)

ntokens = len(dictionary)
criterion = nn.CrossEntropyLoss(size_average=False)

val_data  = list(read(args.test_data, eval_batch_size))

if args.val== True:
    eval_data= val_data
else:
    eval_data=test_data
train_data = batchify(corpus.train_small, args.batch_size)

lr = args.lr
lamb = args.lamb
epsilon = args.epsilon

#change batch size to 1 for dynamic eval
args.batch_size=1
if not(args.grid or args.gridfast):
    print('# running dynamic evaluation')
    #apply dynamic evaluation
    loss = evaluate()
    print('perplexity loss: ' + str(loss[0]))
