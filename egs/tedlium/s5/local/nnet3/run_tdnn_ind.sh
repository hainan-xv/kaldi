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
pnormi=3000
pnormo=300
extra_layer=false
last_factor=1

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

multidir=$1
virtualdir=$2
num_outputs=$3
train_stage=$4

dir=${dir}_${pnormo}_${pnormi}

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
     /export/b0{1,2,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

#  lr1=`echo 0.0015 / $num_outputs | bc -l`
#  lr2=`echo 0.00015 / $num_outputs | bc -l`
  lr1=0.0015
  lr2=0.00015

  steps/nnet3/train_tdnn_ind.sh --stage $train_stage \
    --online-ivector-dir exp/nnet3/ivectors_train \
    --num-outputs $num_outputs \
    --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 0" \
    --feat-type raw \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $lr1 --final-effective-lrate $lr2 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim $pnormi \
    --pnorm-output-dim $pnormo \
    data/train_hires data/lang $multidir/tree $dir  || exit 1;
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
    graph_dir=$virtualdir/graph
  for year in test dev; do
(      steps/nnet3/decode_multi.sh --nj 8 --cmd "$decode_cmd" \
       --num-outputs $num_outputs \
       --online-ivector-dir exp/nnet3/ivectors_$year \
       $graph_dir data/${year}_hires \
       $virtualdir \
       $dir/decode_${year} || exit 1; ) &
  done
  wait
fi

