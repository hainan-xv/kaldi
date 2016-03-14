#!/bin/bash

. path.sh
. cmd.sh

gmm=true
method=multi # joint for joint training; multi for multi-output training
gmm_decode=false
dnn_stage=-100
stage=8
num_trees=2
echo "$0 $@"

. ./utils/parse_options.sh || exit 1;

#set -e

num_leaves=$1
num_gauss=$2


data=data/train
lang=data/lang
alidir=exp/tri3b_ali
dir=exp/Random_${num_trees}/tri_${num_leaves}_${num_gauss}

mkdir -p $dir

if [ "$gmm" == "true" ]; then
  echo training GMM systems
  steps/train_sat_random.sh --cmd "$train_cmd" \
      --num-iters 2 \
      --numtrees $num_trees \
      $num_leaves $num_gauss $data $lang $alidir $dir

  for i in `seq 0 $[$num_trees-1]`; do
    cp $dir/tree_$i/final.mdl $dir/model-$i
  done

  steps/build_virtual_tree.sh --cmd "$train_cmd" --numtrees $num_trees \
      $data $lang $alidir $dir $dir/virtual

  utils/mkgraph.sh data/lang $dir/virtual $dir/virtual/graph

fi

#nnet3dir=${dir}/../tdnn_${num_leaves}
nnet3dir=exp/Random_$num_trees/${method}_tdnn_${num_leaves}
#dnn_stage=81

num_trees=$[$num_trees_L+${num_trees_T}+${num_trees_R}]
#f=$(echo "sqrt ( $num_trees )" | bc -l)
#o=`echo "$f" | awk '{print int($1*350)}'`
#i=$[10*$o]
#echo $o and $i
./local/nnet3/run_tdnn_$method.sh --last-factor $num_trees --extra-layer true --stage $stage --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage
exit
#./local/nnet3/run_tdnn.sh --alidir $dir/tree_0/ --stage $stage --dir ${nnet3dir}_010 $dir $dir/virtual $num_trees $dnn_stage
