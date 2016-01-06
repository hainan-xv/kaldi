#!/bin/bash

. path.sh
. cmd.sh

gmm=true
gmm_decode=false
dnn_stage=-100
echo "$0 $@"

. ./utils/parse_options.sh || exit 1;

#set -e

num_leaves=$1
num_gauss=$2
num_questions=$3

data=data/train_si284
lang=data/lang
alidir=exp/tri4b_ali_si284
dir=exp/expanded_${num_questions}/tri_${num_leaves}_${num_gauss}

set -e

if [ "$gmm" == "true" ]; then
  echo training GMM systems
  steps/build_tree_expand.sh --cmd "$train_cmd" \
      --stage 0 \
      --num-iters 2 --num-questions ${num_questions} \
      $num_leaves $num_gauss $data $lang $alidir $dir $dir/virtual

  num_trees=`ls $dir/ | grep tree_ | wc -l | awk '{print$1}'`

  for i in `seq 0 $[$num_trees-1]`; do
    cp $dir/tree_$i/final.mdl $dir/model-$i
  done

  steps/build_virtual_tree.sh --cmd "$train_cmd" --numtrees $num_trees \
      $data $lang $alidir $dir $dir/virtual

  utils/mkgraph.sh data/lang_test_bd_tgpr $dir/virtual $dir/virtual/graph_bd_tgpr
fi

nnet3dir=${dir}/../tdnn_joint_${num_leaves}_${num_questions}

./local/nnet3/run_tdnn_joint.sh --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage
