#!/bin/bash

# unrescored baseline
# %WER 24.2 | 13098 94476 | 79.2 12.0 8.8 3.4 24.2 57.4 | -0.157 | exp/ihm/nnet3/tdnn_sp/decode_dev/ascore_12/dev_hires.ctm.filt.sys
# %WER 25.4 | 12643 89973 | 77.7 14.0 8.3 3.1 25.4 56.5 | -0.144 | exp/ihm/nnet3/tdnn_sp/decode_eval/ascore_11/eval_hires.ctm.filt.sys

# rescored numbers
# %WER 23.6 | 13098 94467 | 79.5 11.7 8.8 3.1 23.6 56.9 | -0.351 | exp/ihm/nnet3/tdnn_sp/decode_dev_tfrnnlm_3gram/ascore_12/dev_hires.ctm.filt.sys
# %WER 23.8 | 13098 94475 | 79.5 11.9 8.5 3.3 23.8 57.0 | -0.263 | exp/ihm/nnet3/tdnn_sp/decode_dev_tfrnnlm_3gram_unpruned/ascore_11/dev_hires.ctm.filt.sys
# %WER 24.5 | 12643 89976 | 78.4 13.4 8.2 2.9 24.5 55.7 | -0.315 | exp/ihm/nnet3/tdnn_sp/decode_eval_tfrnnlm_3gram/ascore_11/eval_hires.ctm.filt.sys
# %WER 24.7 | 12643 89979 | 78.2 13.5 8.3 2.9 24.7 55.9 | -0.182 | exp/ihm/nnet3/tdnn_sp/decode_eval_tfrnnlm_3gram_unpruned/ascore_11/eval_hires.ctm.filt.sys

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

dir=data/tensorflow_lstm_fast
mkdir -p $dir

steps/tfrnnlm/check_tensorflow_installed.sh

if [ $stage -le 1 ]; then
  local/tfrnnlm/rnnlm_data_prep.sh $dir
fi

if [ $stage -le 2 ]; then
  $cuda_cmd $dir/train_rnnlm.log utils/parallel/limit_num_gpus.sh \
    python -u steps/tfrnnlm/train_lstm_fast.py --data-path=$dir --save-path=$dir/rnnlm --vocab-path=$dir/wordlist.rnn.final
fi

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

if [ $stage -le 3 ]; then
  for decode_set in dev eval; do
    basedir=exp/$mic/nnet3/tdnn_sp/
    decode_dir=${basedir}/decode_${decode_set}

# pruned lattice rescoring
    steps/tfrnnlm/lmrescore_rnnlm_lat_pruned.sh \
      --cmd "$tfrnnlm_cmd --mem 4G" \
      --weight $weight --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/$mic/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_tfrnnlm_${ngram_order}gram  &

# Lattice rescoring, unpruned (slow) version, unparallel
    steps/tfrnnlm/lmrescore_rnnlm_lat.sh \
      --cmd "$tfrnnlm_cmd --mem 4G" \
      --weight $weight --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/$mic/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_tfrnnlm_${ngram_order}gram_unpruned  &

  done
fi

wait
