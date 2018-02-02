#!/bin/bash

# unrescored baseline
# %WER 24.2 | 13098 94476 | 79.2 12.0 8.8 3.4 24.2 57.4 | -0.157 | exp/ihm/nnet3/tdnn_sp/decode_dev/ascore_12/dev_hires.ctm.filt.sys
# %WER 25.4 | 12643 89973 | 77.7 14.0 8.3 3.1 25.4 56.5 | -0.144 | exp/ihm/nnet3/tdnn_sp/decode_eval/ascore_11/eval_hires.ctm.filt.sys

# rescored numbers
# %WER 23.7 | 13098 94474 | 79.7 11.9 8.4 3.4 23.7 57.0 | -0.270 | exp/ihm/nnet3/tdnn_sp/decode_dev_lat_lstm_3gram_unpruned/ascore_11/dev_hires.ctm.filt.sys
# %WER 24.6 | 12643 89982 | 78.4 13.4 8.2 3.0 24.6 55.7 | -0.188 | exp/ihm/nnet3/tdnn_sp/decode_eval_lat_lstm_3gram_unpruned/ascore_11/eval_hires.ctm.filt.sys

mic=ihm
ngram_order=3 # this option when used, the rescoring binary makes an approximation
    # to merge the states of the FST generated from RNNLM. e.g. if ngram-order = 4
    # then any history that shares last 3 words would be merged into one state
stage=1
weight=0.5   # when we do lattice-rescoring, instead of replacing the lm-weights
    # in the lattice with RNNLM weights, we usually do a linear combination of
    # the 2 and the $weight variable indicates the weight for the RNNLM scores

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

dir=data/tensorflow_lstm
mkdir -p $dir

steps/tfrnnlm/check_tensorflow_installed.sh

if [ $stage -le 1 ]; then
  local/tfrnnlm/rnnlm_data_prep.sh $dir
fi

mkdir -p $dir
if [ $stage -le 2 ]; then
  $cuda_cmd $dir/train_rnnlm.log utils/parallel/limit_num_gpus.sh \
    python -u steps/tfrnnlm/train_lstm.py --data-path=$dir --save-path=$dir/rnnlm --vocab-path=$dir/wordlist.rnn.final
fi

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

if [ $stage -le 3 ]; then
  for decode_set in dev eval; do
    basedir=exp/$mic/nnet3/tdnn_sp/
    decode_dir=${basedir}/decode_${decode_set}

    # Lattice rescoring
    steps/tfrnnlm/lmrescore_rnnlm_lat.sh \
      --cmd "$tfrnnlm_cmd --mem 4G" \
      --weight $weight --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/$mic/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_lat_lstm_${ngram_order}gram_unpruned  &

  done
fi

wait
