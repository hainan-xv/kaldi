set -uo pipefail

# configs for 'chain'
affix=
stage=9 # After running the entire script once, you can set stage=12 to tune the neural net only.
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn
decode_iter=

srand=1
chunk_width=140,100,160
chunk_left_context=0
chunk_right_context=0
common_egs_dir=
reporting_email=

# TDNN options
# this script uses the new tdnn config generator so it needs a final 0 to reflect that the final layer input has no splicing
self_repair_scale=0.00001
# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=8
minibatch_size=128
relu_dim=425
frames_per_eg=150
remove_egs=false
xent_regularize=0.1
backstitch_scale=0.2

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=${dir}${affix}_$backstitch_scale

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 9" if you have already
# run those things.

gmm_dir=exp/tri5a
ali_dir=exp/tri5a_ali_sp
lats_dir=${ali_dir/ali/lats} # note, this is a search-and-replace from 'ali' to 'lats'
treedir=exp/chain/tri5a_tree
lang=data/lang_chain

mkdir -p $dir

local/nnet3/run_ivector_common.sh --stage $stage \
  --generate-alignments true || exit 1;

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat ${ali_dir}/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/train_sp \
    data/lang $gmm_dir $lats_dir
  rm ${lats_dir}/fsts.*.gz # save space
fi

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 4000 data/train_sp $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
#  repair_opts=${self_repair_scale:+" --self-repair-scale-nonlinearity $self_repair_scale "}
#
#  steps/nnet3/tdnn/make_configs.py \
#    $repair_opts \
#    --feat-dir data/train_sp_hires \
#    --ivector-dir exp/nnet3/ivectors_train_sp \
#    --tree-dir $treedir \
#    --relu-dim $relu_dim \
#    --splice-indexes "-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0" \
#    --use-presoftmax-prior-scale false \
#    --xent-regularize $xent_regularize \
#    --xent-separate-forward-affine true \
#    --include-log-softmax false \
#    --final-layer-normalize-target $final_layer_normalize_target \
#    $dir/configs || exit 1;

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=200
  relu-renorm-layer name=tdnn2 dim=200 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn3 dim=200 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn4 dim=200 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn5 dim=200 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn6 dim=200 input=Append(-6,-3,0)

  ## adding the layers for chain branch
  relu-renorm-layer name=prefinal-chain dim=512 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-renorm-layer name=prefinal-xent input=tdnn6 dim=512 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
     # spread the egs over various machines.  will help reduce overload of any
     # one machine.
     utils/create_split_dir.pl /export/b{09,10,11,12}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  touch $dir/egs/.nodelete

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --trainer.optimization.backstitch-training-scale $backstitch_scale \
    --feat.online-ivector-dir=exp/nnet3/ivectors_train_sp \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.00005 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=4 \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=5 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.optimization.proportional-shrink=60.0 \
    --trainer.num-chunk-per-minibatch=256,128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=data/train_sp_hires \
    --tree-dir=$treedir \
    --lat-dir=$lats_dir \
    --dir=$dir  || exit 1;

# steps/nnet3/chain/train.py --stage $train_stage \
#   --cmd "$decode_cmd" \
#   --trainer.optimization.backstitch-training-scale $backstitch_scale \
#   --feat.online-ivector-dir exp/nnet3/ivectors_train_sp \
#   --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
#   --chain.xent-regularize $xent_regularize \
#   --chain.leaky-hmm-coefficient 0.1 \
#   --chain.l2-regularize 0.00005 \
#   --chain.apply-deriv-weights false \
#   --chain.lm-opts="--num-extra-lm-states=2000" \
#   --egs.stage $get_egs_stage \
#   --egs.opts "--frames-overlap-per-eg 0" \
#   --egs.chunk-width $frames_per_eg \
#   --trainer.num-chunk-per-minibatch $minibatch_size \
#   --trainer.frames-per-iter 1500000 \
#   --trainer.num-epochs $num_epochs \
#   --trainer.optimization.num-jobs-initial $num_jobs_initial \
#   --trainer.optimization.num-jobs-final $num_jobs_final \
#   --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
#   --trainer.optimization.final-effective-lrate $final_effective_lrate \
#   --trainer.max-param-change $max_param_change \
#   --cleanup.remove-egs $remove_egs \
#   --cleanup.preserve-model-interval 20 \
#   --feat-dir data/train_sp_hires \
#   --tree-dir $treedir \
#   --lat-dir $lats_dir \
#   --dir $dir || exit 1;
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi

  for decode_set in dev test; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --stage 3 \
      --nj $(wc -l < data/$decode_set/spk2utt) --cmd "$decode_cmd" $iter_opts \
      --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
      --scoring-opts "--min_lmwt 5 --max_lmwt 15" \
      $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter} || exit 1;

  done
fi

wait
