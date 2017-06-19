#!/bin/bash
mic=ihm
ngram_order=3
model_type=small
stage=1
weight=0.5

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

dir=data/new_tensorflow/$model_type
mkdir -p $dir

if [ $stage -le 1 ]; then
  local/tensorflow/train_rnnlm.sh $dir
fi

if [ $stage -le 2 ]; then
  mkdir -p $dir/
  python local/tensorflow/rnnlm.py --data_path=$dir --model=$model_type --save_path=$dir/rnnlm --wordlist_save_path=$dir/wordlist.rnn.final
fi

has_oos=`grep "<oos>" $dir/wordlist.rnn.final | wc -l | awk '{print $1}'`
if [ $has_oos == "0" ]; then
  n=`wc -l $dir/wordlist.rnn.final | awk '{print $1}'`
  echo n is $n
  echo "<oos> $n" >> $dir/wordlist.rnn.final
fi

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

date
if [ $stage -le 3 ]; then
#  for decode_set in dev; do
  for decode_set in dev eval; do
    basedir=exp/$mic/nnet3/tdnn_sp/
    decode_dir=${basedir}/decode_${decode_set}

    # Lattice rescoring
    steps/lmrescore_rnnlm_lat.sh \
      --cmd "$tensorflow_cmd --mem 16G" \
      --rnnlm-ver tensorflow  --weight $weight --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/$mic/${decode_set}_hires ${decode_dir} \
      ${decode_dir}.new.tfrnnlm.lat.${ngram_order}gram.$weight  &

  done
fi

wait
date
