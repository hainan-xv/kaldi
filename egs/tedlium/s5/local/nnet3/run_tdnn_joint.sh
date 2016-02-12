#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
dir=
expand=false

pnormi=3500
pnormo=350

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh
multidir=$1
virtualdir=$2
num_outputs=$3
train_stage=$4

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
     /export/b0{1,2,5,6}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

#  train_stage=`ls $dir | grep .mdl | sed s=.mdl==g | sort -n | tail -n 1`
  echo train_stage $train_stage

  steps/nnet3/train_tdnn_joint.sh --stage $train_stage \
    --num-outputs $num_outputs \
    --num-epochs 8 --num-jobs-initial 2 --num-jobs-final 14 \
    --splice-indexes "-4,-3,-2,-1,0,1,2,3,4  0  -2,2  0  -4,4 0" \
    --feat-type raw \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim $pnormi \
    --pnorm-output-dim $pnormo \
    --expand $expand \
    --tree-mapping $virtualdir/tree-mapping \
    data/train data/lang $multidir/tree $virtualdir/ $dir  || exit 1;
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
    graph_dir=$virtualdir/graph
    # use already-built graphs.
  for year in test dev; do
(      steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" \
       $graph_dir data/${year} \
       $dir/decode_${year} || exit 1; ) &
  done
  wait
  echo done
fi

