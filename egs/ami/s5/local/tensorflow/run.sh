#!/bin/bash
mic=ihm
ngram_order=4
model_type=small
dir=data/tensorflow
stage=1

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

mkdir -p $dir

if [ $stage -le 1 ]; then
  local/tensorflow/train_rnnlm.sh $dir
fi

if [ $stage -le 2 ]; then
#  python local/tensorflow/rnnlm.py --data_path=$dir --model=test --save_path=$dir/rnnlm --wordlist_save_path=$dir/wordlist.rnn.final
#  python local/tensorflow/rnnlm.py --data_path=$dir --model=small --save_path=$dir/rnnlm --wordlist_save_path=$dir/wordlist.rnn.final
python local/tensorflow/rnnlm.py --data_path=$dir --model=medium --save_path=$dir/rnnlm --wordlist_save_path=$dir/wordlist.rnn.final
#python local/tensorflow/rnnlm.py --data_path=$dir --model=large --save_path=$dir/model.large
fi

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

if [ $stage -le 3 ]; then
#  for decode_set in dev; do
  for decode_set in dev eval; do
    basedir=exp/$mic/nnet3/tdnn_sp/
    decode_dir=${basedir}/decode_${decode_set}

    # Lattice rescoring
    steps/lmrescore_rnnlm_lat.sh \
      --cmd "$decode_cmd --mem 16G" \
      --rnnlm-ver tensorflow  --weight 0.5 --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/$mic/${decode_set}_hires ${decode_dir} \
      ${decode_dir}.tfrnnlm.lat.${ngram_order}gram &

  done
fi
