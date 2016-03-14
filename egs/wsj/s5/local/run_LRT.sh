#!/bin/bash

. path.sh
. cmd.sh

gmm=true
method=multi # joint for joint training; multi for multi-output training
gmm_decode=false
stage=8
dnn_stage=-100
extra_layer=false
pi=2000
po=250

echo "$0 $@"

. ./utils/parse_options.sh || exit 1;

#set -e

num_trees_L=$1
num_trees_T=$2
num_trees_R=$3
lambda=$4
num_leaves=$5
num_gauss=$6

num_trees=$[$num_trees_L+$num_trees_T+$num_trees_L]

data=data/train_si284
lang=data/lang
alidir=exp/tri4b_ali_si284
dir=exp/LRT_${num_trees_L}_${num_trees_T}_${num_trees_R}_$lambda/tri_${num_leaves}_${num_gauss}

mkdir -p $dir

if [ "$gmm" == "true" ]; then
  echo training GMM systems
  steps/train_sat_LRT.sh --cmd "$train_cmd" \
      --numtrees_L $num_trees_L \
      --numtrees_T $num_trees_T \
      --numtrees_R $num_trees_R \
      --lambda $lambda \
      --num-iters 2 \
      $num_leaves $num_gauss $data $lang $alidir $dir

  for i in `seq 0 $[$num_trees-1]`; do
    cp $dir/tree_$i/final.mdl $dir/model-$i
  done

  steps/build_virtual_tree.sh --cmd "$train_cmd" --numtrees $num_trees \
      $data $lang $alidir $dir $dir/virtual

  utils/mkgraph.sh data/lang_test_bd_tgpr $dir/virtual $dir/virtual/graph_bd_tgpr

fi

nnet3dir=${dir}/../${method}_tdnn_${num_leaves}_${pi}_${po}

if [ "$extra_layer" == "true" ]; then
  nnet3dir=${nnet3dir}_extra
fi

#dnn_stage=81
#./local/nnet3/run_tdnn_$method.sh --pi $pi --po $po --extra-layer $extra_layer --stage $stage --train-stage $dnn_stage --dir $nnet3dir $dir $dir/virtual $num_trees 
nnet3dir=${nnet3dir}_baseline
./local/nnet3/run_tdnn.sh --alidir $dir/tree_0/ --stage $stage --train-stage $dnn_stage --dir $nnet3dir $dir $dir/virtual $num_trees 
