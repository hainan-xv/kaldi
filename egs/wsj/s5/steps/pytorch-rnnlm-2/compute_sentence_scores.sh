#!/bin/bash

# This script is very similar to utils/rnnlm_compute_scores.sh, and it computes
# log-likelihoods from a Kaldi-RNNLM model instead of that of Mikolov's RNNLM.
# Because Kaldi-RNNLM uses letter-features which does not need an <OOS> symbol,
# we don't need the "unk.probs" file any more to add as a penalty term in sentence
# likelihoods.

. ./path.sh || exit 1;
. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: $0 <rnn-dir> <temp-dir> <input-text> <output-scores>"
  exit 1;
fi

dir=$1
tempdir=$2
text_in=$3
scores_out=$4

mkdir -p $tempdir
cat $text_in | cut -d " " -f1  > $tempdir/utt_ids
cat $text_in | cut -d " " -f2- > $tempdir/text.txt

echo /home/tongfei/app/anaconda/bin/python -u steps/pytorch-rnnlm-2/rnnlm_eval.py --MODEL data/new-pytorch/best.mdl --TEST $tempdir/text.txt --VOCAB data/pytorch-lm-2/vocab.txt
/home/tongfei/app/anaconda/bin/python -u steps/pytorch-rnnlm-2/dynamic_eval.py --MODEL data/new-pytorch/epoch.19.mdl --TEST $tempdir/text.txt --VOCAB data/pytorch-lm-2/vocab.txt | grep score | awk '{print $4}' > $tempdir/loglikes.rnn
#/home/tongfei/app/anaconda/bin/python -u steps/pytorch-rnnlm-2/rnnlm_eval.py --MODEL data/new-pytorch/epoch.19.mdl --TEST $tempdir/text.txt --VOCAB data/pytorch-lm-2/vocab.txt | grep score | awk '{print $4}' > $tempdir/loglikes.rnn

[ $(cat $tempdir/loglikes.rnn | wc -l) -ne $(cat $tempdir/text.txt | wc -l) ] && \
  echo "$0: rnnlm rescoring failed" && exit 1;

# We need the negative log-probabilities
paste $tempdir/utt_ids $tempdir/loglikes.rnn >$scores_out


