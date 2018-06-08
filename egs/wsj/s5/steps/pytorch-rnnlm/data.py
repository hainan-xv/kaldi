import os
import torch


def read_vocab(fname):
    w2i = {}
    i2w = {}
    with open(fname, "r") as f:
        i = 0
        for line in f:
            w2i[line.strip()] = i
            i2w[i] = line.strip()
            i = i + 1
        w2i["<RNN_UNK>"] = i
        i2w[i] = "<RNN_UNK>"
        unk_id = i

    return w2i, i2w, unk_id
        
class Corpus(object):
    def __init__(self, path):
        self.w2i, self.i2w, self.unk_id = read_vocab(os.path.join(path, 'vocab.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                tokens += len(line.split()) + 1

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['</s>']
                for word in words:
                    ids[token] = self.w2i.get(word, self.unk_id)
                    token += 1

        return ids
