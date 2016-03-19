#!/bin/bash

. path.sh
. cmd.sh

gmm=true
method=multi # joint for joint training; multi for multi-output training
gmm_decode=false
dnn_stage=-100
stage=8
extra=false
num_trees=2
prob=0.8

echo "$0 $@"

. ./utils/parse_options.sh || exit 1;

#set -e

num_leaves=$1
num_gauss=$2


data=data/train
lang=data/lang
alidir=exp/tri3_ali
dir=exp/Random_$num_trees/tri_${num_leaves}_${num_gauss}

mkdir -p $dir

if [ "$gmm" == "true" ]; then
  echo training GMM systems
  steps/train_sat_random.sh --cmd "$train_cmd" \
      --numtrees $num_trees \
      --num-iters 2 \
      $num_leaves $num_gauss $data $lang $alidir $dir

  for i in `seq 0 $[$num_trees-1]`; do
    cp $dir/tree_$i/final.mdl $dir/model-$i
  done

  steps/build_virtual_tree.sh --cmd "$train_cmd" --numtrees $num_trees \
      $data $lang $alidir $dir $dir/virtual

  utils/mkgraph.sh data/lang_test $dir/virtual $dir/virtual/graph

fi

#nnet3dir=${dir}/../tdnn_${num_leaves}
nnet3dir=exp/Random_$num_trees/${method}_tdnn_${num_leaves}_$prob
#dnn_stage=81

#num_trees=$[$num_trees_L+${num_trees_T}+${num_trees_R}]
#f=$(echo "sqrt ( $num_trees )" | bc -l)
#o=`echo "$f" | awk '{print int($1*350)}'`
#i=$[10*$o]
#echo $o and $i
#./local/nnet3/run_tdnn_$method.sh --last-factor $num_trees --pnormi 4000 --pnormo 400 --extra-layer $extra --stage $stage --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage
./local/nnet3/run_tdnn_$method.sh --extra-layer $extra --stage $stage --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage
