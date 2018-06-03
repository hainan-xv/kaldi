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

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--MB_SIZE', type=int, help='minibatch size')
parser.add_argument('--EMBED_SIZE', type=int, help='embedding size')
parser.add_argument('--HIDDEN_SIZE', type=int, help='hidden size')
parser.add_argument('--NUM_LAYERS', type=int, help='number of layers', default=2)
parser.add_argument('--DROPOUT', type=float, help='dropout', default=0)
parser.add_argument('--EPOCHS', type=int, help='number of epochs')
parser.add_argument('--TRAIN', type=str, help='train text')
parser.add_argument('--VALID', type=str, help='valid text')
parser.add_argument('--VOCAB', type=str, help='vocab text')
parser.add_argument('--CUDA', action='store_true', help='use CUDA')
parser.add_argument('--gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA")
args = parser.parse_args()

torch.cuda.set_device(args.gpus[0])

train_file = args.TRAIN
valid_file = args.VALID
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
  w2i["<MASK>"] = 0
  with open(fname, "r") as f:
    i = 1
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
      sent.insert(0, w2i["<s>"])
      sent.append(w2i["</s>"])
      yield torch.LongTensor(sent)

def get_batch(sequences, volatile=False):
  lengths = torch.LongTensor([len(s) for s in sequences])
  batch   = torch.LongTensor(lengths.max(), len(sequences)).fill_(mask)
  for i, s in enumerate(sequences):
    batch[:len(s), i] = s
  if args.CUDA:
    batch = batch.cuda()
  return Variable(batch, volatile=volatile), lengths

w2i, unk_index = read_vocab(vocab_file)
mask = w2i['<MASK>']
assert mask == 0
train = list(read(train_file, unk_index))
valid = list(read(valid_file, unk_index))
vocab_size = len(w2i)
BOS = w2i['<s>']
EOS = w2i['</s>']

print ("vocab size is ", vocab_size)

# build the model
rnnlm = RNNModel("LSTM", vocab_size, args.EMBED_SIZE, args.HIDDEN_SIZE, args.NUM_LAYERS, args.DROPOUT)
#parameters = list(filter(lambda p: p.requires_grad, rnnlm.parameters()))
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, rnnlm.parameters()), lr=0.1)
optimizer = optim.Adam(rnnlm.parameters(), lr=0.1)
#optimizer = optim.SGD(rnnlm.parameters(), lr=201)
weight = torch.FloatTensor(vocab_size).fill_(1)
weight[mask] = 0
loss_fn = nn.CrossEntropyLoss(weight, size_average=False)
#loss_fn = nn.CrossEntropyLoss(size_average=False, ignore_index=mask)

if args.CUDA:
  rnnlm.cuda()
  loss_fn.cuda()

train_order = range(0, len(train), args.MB_SIZE)
valid_order = range(0, len(valid), args.MB_SIZE)

# Perform training
print("startup time: %r" % (time.time() - start))
start = time.time()
i = total_time = dev_time = total_tagged = current_words = current_loss = 0

file_id = 0
for ITER in range(args.EPOCHS):
#  random.shuffle(train)
  i = 0
  print ("starting epoch", ITER)

  for sid in train_order:
    rnnlm.train()
#    print (i)
    i += 1
    # train

    train_subset = train[sid:sid + args.MB_SIZE]
    train_subset.sort(key=lambda x:-len(x))
    batch, lengths = get_batch(train_subset)

#    print (lengths)

    torch.nn.utils.rnn.pack_padded_sequence(batch, [len(s) for s in train_subset], batch_first=False)

    hidden = rnnlm.init_hidden(lengths.size(0))
#    print (batch)
    output, h = rnnlm(batch[:-1], hidden)
    scores = output.view(-1, vocab_size)
    loss = loss_fn(scores, batch[1:].view(-1)) / (lengths.sum() - lengths.size(0))
    # optimization
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm(rnnlm.parameters(), 0.25)

    optimizer.step()
    # log loss
    current_words += lengths.sum() - lengths.size(0)  # ignore <s>
    current_loss += loss.data[0] * (lengths.sum() - lengths.size(0))

    if i % int(40000 / args.MB_SIZE) == 0:
      print(" training %.1f%% train PPL=%.4f" % (i / len(train_order) * 100, np.exp(current_loss / current_words)))
      total_tagged += current_words
      current_loss = current_words = 0
      total_time = time.time() - start
    # log perplexity
    if i % int(40000 / args.MB_SIZE) == 0:
      rnnlm.eval()
      dev_start = time.time()
      dev_loss = dev_words = 0
      for j in valid_order:
        batch, lengths = get_batch(valid[j:j + args.MB_SIZE], volatile=True)
        hidden = rnnlm.init_hidden(lengths.size(0))
        output, h = rnnlm(batch[:-1], hidden)
        scores = output.view(-1, vocab_size)
        dev_loss += loss_fn(scores, batch[1:].view(-1)).data[0]
        dev_words += lengths.sum() - lengths.size(0)  # ignore <s>
      dev_time += time.time() - dev_start
      train_time = time.time() - start - dev_time
#      print("           dev   PPL=%.4f word_per_sec=%.4f" % (
#          np.exp(dev_loss / dev_words), total_tagged / train_time))
      print("  nll=%.4f, ppl=%.4f, words=%r, time=%.4f, word_per_sec=%.4f" % (
          dev_loss / dev_words, np.exp(dev_loss / dev_words), dev_words, train_time, total_tagged / train_time))

      file_id = file_id + 1
      with open("data/pytorch-lm-2/tmp.%r.mdl" % file_id, 'wb') as f:
        torch.save(rnnlm, f)

  with open("data/pytorch-lm-2/epoch.%r.mdl" % ITER, 'wb') as f:
    torch.save(rnnlm, f)

  print("epoch %r finished" % ITER)
