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

cat $tempdir/text.txt | awk -v voc=$dir/vocab.txt -v unk=$dir/unk.probs \
  -v logprobs=$tempdir/loglikes.oov \
 'BEGIN{ while((getline<voc)>0) { invoc[$1]=1; } while ((getline<unk)>0){ unkprob[$1]=$2;} }
  { logprob=0;
    if (NF==0) { printf "<RNN_UNK>"; logprob = log(1.0e-07);
      print "Warning: empty sequence." | "cat 1>&2"; }
    for (x=1;x<=NF;x++) { w=$x;  
    if (invoc[w]) { printf("%s ",w); } else {
      printf("<RNN_UNK> ");
      if (unkprob[w] != 0) { logprob += log(unkprob[w]); }
      else { print "Warning: unknown word ", w | "cat 1>&2"; logprob += log(1.0e-07); }}}
    printf("\n"); print logprob > logprobs } ' > $tempdir/text.nounk

/home/tongfei/app/anaconda/bin/python -u steps/pytorch-rnnlm/rnnlm_eval.py --MODEL $dir/best.mdl --TEST $tempdir/text.nounk --VOCAB $dir/vocab.txt | grep score | awk '{print $4}' > $tempdir/loglikes.rnn

[ $(cat $tempdir/loglikes.rnn | wc -l) -ne $(cat $tempdir/text.txt | wc -l) ] && \
  echo "$0: rnnlm rescoring failed" && exit 1;

paste $tempdir/loglikes.rnn $tempdir/loglikes.oov | awk '{print ($1-$2);}' >$tempdir/scores

# We need the negative log-probabilities
paste $tempdir/utt_ids $tempdir/scores >$scores_out


