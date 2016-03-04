#!/bin/bash

. path.sh
. cmd.sh

gmm=true
method=multi # joint for joint training; multi for multi-output training
gmm_decode=true
dnn_stage=-100
mic=sdm1
sp=false
extra_layer=false
realign=false

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

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

data=data/$mic/train
lang=data/lang
alidir=exp/$mic/tri4a_ali
dir=exp/$mic/LRT_${num_trees_L}_${num_trees_T}_${num_trees_R}_$lambda/tri_${num_leaves}_${num_gauss}

if [ "$realign" == "true" ]; then
  nj=30
  steps/train_sat.sh  --cmd "$train_cmd" \
    5000 80000 data/$mic/train data/lang exp/$mic/tri3a_ali exp/$mic/tri4a
  # Decode,
  graph_dir=exp/$mic/tri4a/graph_${LM}
  $highmem_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/$mic/tri4a $graph_dir
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd"  --config conf/decode.conf \
    $graph_dir data/$mic/dev exp/$mic/tri4a/decode_dev_${LM} &
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/eval exp/$mic/tri4a/decode_eval_${LM} &

  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train data/lang exp/$mic/tri4a exp/$mic/tri4a_ali
fi

if [ "$gmm" == "true" ]; then
  echo training GMM systems
  steps/train_sat_LRT.sh --cmd "$train_cmd" \
      --num-iters 2 \
      --numtrees_L $num_trees_L \
      --numtrees_T $num_trees_T \
      --numtrees_R $num_trees_R \
      --lambda $lambda \
      $num_leaves $num_gauss $data $lang $alidir $dir

  for i in `seq 0 $[$num_trees-1]`; do
    cp $dir/tree_$i/final.mdl $dir/model-$i
  done

  if [ "$sp" == "true" ]; then
    alidir=exp/$mic/tri4a_sp_ali
  fi

  steps/build_virtual_tree.sh --cmd "$train_cmd" --numtrees $num_trees \
      $data $lang $alidir $dir $dir/virtual

  $highmem_cmd $dir/virtual/graph_$LM/mkgraph.log utils/mkgraph.sh ${lang}_${LM} $dir/virtual $dir/virtual/graph_$LM
fi

nnet3dir=${dir}/../${method}_tdnn_${num_leaves}

./local/nnet3/run_tdnn_$method.sh --mic $mic --dir $nnet3dir $dir $dir/virtual $num_trees $dnn_stage
