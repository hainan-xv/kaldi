#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
pnormi=3500
pnormo=350

dir=exp/nnet3/nnet_tdnn_a_${pnormi}_$pnormo
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

#local/nnet3/run_ivector_common.sh --stage $stage || exit 1;

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

#    --feat-type raw \
#    --cmvn-opts "--norm-means=false --norm-vars=false" \
  steps/nnet3/train_tdnn.sh --stage $train_stage \
    --feat-type lda \
    --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
    --splice-indexes "-4,-3,-2,-1,0,1,2,3,4  0  -2,2  0  -4,4 0" \
    --initial-effective-lrate 0.01 --final-effective-lrate 0.001 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim $pnormi \
    --pnorm-output-dim $pnormo \
    data/train_clean_100 data/lang exp/tri4b_ali_clean_100 $dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
#for test in test_clean test_other dev_clean dev_other; do
  for test in test_clean test_other dev_clean dev_other; do
    (graph_dir=exp/tri4b/graph_tgsmall
    # use already-built graphs.
    steps/nnet3/decode.sh --nj 20 --cmd "$decode_cmd" \
      --transform-dir exp/tri4b/decode_tgsmall_$test \
      $graph_dir data/$test $dir/decode_tgsmall_$test

    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test $dir/decode_{tgsmall,tglarge}_$test ) &

  done
  wait
fi
