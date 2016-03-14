#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=-100
train_stage=-100
#dir=exp/nnet3/nnet_tdnn_multi_$4
dir=
pnormi=2400
pnormo=400
extra_layer=false
last_factor=1

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

multidir=$1
virtualdir=$2
num_outputs=$3
train_stage=$4

if [ "$extra_layer" == "true" ]; then
  dir=${dir}_extra
fi

if [ "$last_factor" != "1" ]; then
  dir=${dir}_enlarge$last_factor
fi

echo dir is $dir

if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{1,2,5,4}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

#    --cleanup false \

  steps/nnet3/train_tdnn_ind.sh --stage $train_stage \
    --num-outputs $num_outputs \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_train_clean_100 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 0" \
    --initial-effective-lrate 0.015 --final-effective-lrate 0.0015 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim $pnormi \
    --pnorm-output-dim $pnormo \
    data/train_clean_100_hires data/lang $multidir/tree $dir  || exit 1;
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.

  for test in test_clean test_other dev_clean dev_other; do
(    graph_dir=$virtualdir/graph_tgsmall
    # use already-built graphs.
    steps/nnet3/decode_multi.sh --nj 20 --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3/ivectors_$test \
      --num-outputs $num_outputs \
      $graph_dir data/${test}_hires $virtualdir $dir/decode_tgsmall_$test

    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test $dir/decode_{tgsmall,tglarge}_$test || exit 1;
      ) &
  done
  wait
fi

