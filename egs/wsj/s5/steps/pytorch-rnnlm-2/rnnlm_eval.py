from __future__ import division, print_function
import time
import sys
import random
import argparse
from itertools import count
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import model

parser = argparse.ArgumentParser()
parser.add_argument('--MB_SIZE', type=int, help='minibatch size', default=40)
parser.add_argument('--TEST', type=str, help='test text')
parser.add_argument('--VOCAB', type=str, help='vocab text')
parser.add_argument('--MODEL', type=str, help='model location')
parser.add_argument('--CUDA', action='store_true', help='use CUDA')
parser.add_argument('--gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA")

args = parser.parse_args()

if args.CUDA:
  torch.cuda.set_device(args.gpus[0])

test_file = args.TEST
vocab_file = args.VOCAB

class RNNLM(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
    super(RNNLM, self).__init__()
    self.hidden_size = hidden_size
    self.embeddings = nn.Embedding(vocab_size, embed_size)
    self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
    self.proj = nn.Linear(hidden_size, vocab_size)
    self.proj.weight.data = self.embeddings.weight.data  # tying
    self.embeddings.weight.data.uniform_(-0.1, 0.1)
  def forward(self, sequences):
    rnn_output, _ = self.rnn(self.embeddings(sequences))
    return self.proj(rnn_output.view(-1, self.hidden_size))

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

def read_vocab(fname):
  w2i = {}
#  w2i["<MASK>"] = 0
  with open(fname, "r") as f:
    i = 0
    for line in f:
      w2i[line.strip()] = i
      i = i + 1
  w2i["<RNN_UNK>"] = i
  unk_id = i
  return w2i, unk_id

def read(fname, unk_index):
  """
  Read a file where each line is of the form "word1 word2 ..."
  Yields lists of the form [word1, word2, ...]
  """
  with open(fname, "r") as fh:
    for line in fh:
      sent = [w2i.get(x, unk_index) for x in line.strip().split()]
      sent.insert(0, w2i["</s>"])
      sent.append(w2i["</s>"])
      yield torch.LongTensor(sent)

def get_batch(sequences, volatile=False):
#  print (sequences)
  lengths = torch.LongTensor([len(s) for s in sequences])
  batch   = torch.LongTensor(lengths.max(), len(sequences)).fill_(mask)
  for i, s in enumerate(sequences):
    batch[:len(s), i] = s
  if args.CUDA:
    batch = batch.cuda()
  return Variable(batch, volatile=volatile), lengths

w2i, unk_index = read_vocab(vocab_file)
mask = w2i['<s>']
test = list(read(test_file, unk_index))
vocab_size = len(w2i)
print ("vocab size is ", vocab_size)

weight = torch.FloatTensor(vocab_size).fill_(1)
weight[mask] = 0
loss_fn = nn.CrossEntropyLoss(weight, size_average=False, reduce=False)

test_order = range(0, len(test), args.MB_SIZE)

with open(args.MODEL, 'rb') as f:
    rnnlm = torch.load(f, map_location=lambda storage, loc: storage)

if args.CUDA:
  rnnlm.cuda()
  loss_fn.cuda()

# log perplexity
dev_loss = dev_words = 0
for j in test_order:
  rnnlm.zero_grad()
  batch, lengths = get_batch(test[j:j + args.MB_SIZE], volatile=True)
  hidden = rnnlm.init_hidden(lengths.size(0))
  output, h = rnnlm(batch[:-1], hidden)
  scores = output.view(-1, vocab_size)
#  output = rnnlm(batch[:-1])
#  scores = output.view(-1, vocab_size)
  loss = loss_fn(scores, batch[1:].view(-1))
  loss = loss.view(lengths.max() - 1, -1)
#  loss = loss.view(-1, lengths.max() - 1)
#  loss = torch.t(loss)
  row_sums = torch.sum(loss, dim=0).data
#  row_sums = loss.data
  scores = row_sums.numpy().tolist()
  for i, s in enumerate(scores):
    print ("score", j + i, "is", s)
