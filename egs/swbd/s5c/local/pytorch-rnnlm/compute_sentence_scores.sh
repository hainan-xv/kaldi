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

/home/tongfei/app/anaconda/bin/python -u local/pytorch-rnnlm/dynamiceval.py --model $dir/model.pt --gpu -1 --val --data data/pytorch-lm/ --lr 0 --test_data $tempdir/text.txt | grep sentence | awk '{print $NF}' > $tempdir/loglikes.rnn

[ $(cat $tempdir/loglikes.rnn | wc -l) -ne $(cat $tempdir/text.txt | wc -l) ] && \
  echo "$0: rnnlm rescoring failed" && exit 1;

# We need the negative log-probabilities
paste $tempdir/utt_ids $tempdir/loglikes.rnn >$scores_out


