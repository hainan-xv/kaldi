#!/bin/bash

. path.sh
. cmd.sh

gmm=true
method=multi # joint for joint training; multi for multi-output training
gmm_decode=false
dnn_stage=-100
stage=0
extra=false
factor=1
num_trees=2
prob=0.8
echo "$0 $@"

. ./utils/parse_options.sh || exit 1;

#set -e

num_leaves=$1
num_gauss=$2


data=data/train_clean_100
lang=data/lang
alidir=exp/tri4b_ali_clean_100
dir=exp/Random_$num_trees/tri_${num_leaves}_${num_gauss}_$prob

mkdir -p $dir

if [ "$gmm" == "true" ]; then
  echo training GMM systems
  steps/train_sat_random.sh --cmd "$train_cmd" \
      --prob $prob \
      --numtrees $num_trees \
      --num-iters 2 \
      $num_leaves $num_gauss $data $lang $alidir $dir

  for i in `seq 0 $[$num_trees-1]`; do
    cp $dir/tree_$i/final.mdl $dir/model-$i
  done

  steps/build_virtual_tree.sh --cmd "$train_cmd" --numtrees $num_trees \
      $data $lang $alidir $dir $dir/virtual

  utils/mkgraph.sh data/lang_test_tgsmall $dir/virtual $dir/virtual/graph_tgsmall

fi

nnet3dir=${dir}/../${method}_tdnn_${num_leaves}_$prob

#dnn_stage=81
./local/nnet3/run_tdnn_$method.sh --last-factor $factor --extra-layer $extra --stage $stage --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage
