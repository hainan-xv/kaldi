#!/bin/bash

. path.sh
. cmd.sh

gmm=true
method=joint # joint for joint training; multi for multi-output training
gmm_decode=false
dnn_stage=-100
stage=0
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

data=data/train
lang=data/lang
alidir=exp/tri3_ali
dir=exp/$method/LRT_${num_trees_L}_${num_trees_T}_${num_trees_R}_$lambda/tri_${num_leaves}_${num_gauss}

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

  utils/mkgraph.sh data/lang_test $dir/virtual $dir/virtual/graph

  if [ "$gmm_decode" == "true" ]; then
    echo can not decode multi gmm
    exit 1
    steps/decode_multi.sh --cmd "$decode_cmd" --nj 10 \
        --numtrees $num_trees --transform_dir exp/tri4b/decode_bd_tgpr_dev93 \
        $dir/virtual/graph data/test_dev93 $dir/virtual/decode_dev93 $dir/virtual/tree-mapping &
    steps/decode_multi.sh --cmd "$decode_cmd" --nj 8 \
        --numtrees $num_trees --transform_dir exp/tri4b/decode_bd_tgpr_eval92 \
        $dir/virtual/graph data/test_eval92 $dir/virtual/decode_eval92 $dir/virtual/tree-mapping &
  fi
fi

#nnet3dir=${dir}/../tdnn_${num_leaves}
nnet3dir=exp/$method/LRT_${num_trees_L}_${num_trees_T}_${num_trees_R}_$lambda/tdnn_${num_leaves}
#dnn_stage=81

num_trees=$[$num_trees_L+${num_trees_T}+${num_trees_R}]
f=$(echo "sqrt ( $num_trees )" | bc -l)
o=`echo "$f" | awk '{print int($1*350)}'`
i=$[10*$o]
echo $o and $i
./local/nnet3/run_tdnn_$method.sh --stage $stage --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage
