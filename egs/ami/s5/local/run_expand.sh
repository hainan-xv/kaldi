#!/bin/bash

. path.sh
. cmd.sh

gmm=true
gmm_decode=false
dnn_stage=-100
mic=ihm
echo "$0 $@"

. ./utils/parse_options.sh || exit 1;

#set -e

num_leaves=$1
num_gauss=$2
num_questions=$3

data=data/$mic/train
lang=data/lang
alidir=exp/$mic/tri4a_ali
dir=exp/$mic/expanded_${num_questions}/tri_${num_leaves}_${num_gauss}

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

set -e

if [ "$gmm" == "true" ]; then
  echo training GMM systems
false &&  steps/build_tree_expand.sh --cmd "$train_cmd" \
      --num-iters 2 --num-questions ${num_questions} \
      $num_leaves $num_gauss $data $lang $alidir $dir $dir/virtual

  num_trees=`ls $dir/ | grep tree- | wc -l | awk '{print$1}'`
  echo $num_trees

  for i in `seq 0 $[$num_trees-1]`; do
    cp $dir/tree_$i/final.mdl $dir/model-$i
  done
  steps/build_virtual_tree.sh --cmd "$train_cmd" --numtrees $num_trees \
      --expand true \
      $data $lang $alidir $dir $dir/virtual || exit 1

  $highmem_cmd $dir/virtual/graph_$LM/mkraph.log utils/mkgraph.sh ${lang}_$LM $dir/virtual $dir/virtual/graph_$LM || exit 1
fi

num_trees=`ls $dir/ | grep tree- | wc -l | awk '{print$1}'`
nnet3dir=${dir}/../tdnn_joint_${num_leaves}_${num_questions}

mkdir -p $nnet3dir
cp $dir/virtual/matrix $nnet3dir/

./local/nnet3/run_tdnn_joint.sh --dir $nnet3dir \
    --expand true \
    $dir $dir/virtual $num_trees $dnn_stage
