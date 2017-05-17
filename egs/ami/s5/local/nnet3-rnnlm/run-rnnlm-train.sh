#!/bin/bash

train_text=data/sdm1/train/text
dev_text=data/sdm1/dev/text

cmd=queue.pl

num_words_in=10000
num_words_out=10000

stage=-100
bos="<s>"
eos="</s>"
oos="<oos>"

max_param_change=20
num_iters=160

num_train_frames_combine=10000 # # train frames for the above.                  
num_frames_diagnostic=2000 # number of frames for "compute_prob" jobs  
num_archives=4

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
minibatch_size=1:128

hidden_dim=200
initial_learning_rate=0.08
final_learning_rate=0.004
learning_rate_decline_factor=1.01

# LSTM parameters
num_lstm_layers=1
cell_dim=80
#hidden_dim=200
recurrent_projection_dim=0
non_recurrent_projection_dim=64
norm_based_clipping=true
clipping_threshold=30
label_delay=0  # 5
splice_indexes=0

use_gpu=yes
num_samples=256
momentum=0
wang_scale=0

type=rnn

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

egsdir=data/nnet3_rnnlm_egs
outdir=data/nnet3_rnnlm_${hidden_dim}_${num_samples}_${wang_scale}
srcdir=data/local/dict
set -e

mkdir -p $egsdir
mkdir -p $outdir

if [ $stage -le -4 ]; then
  echo Data Preparation
  cat $srcdir/lexicon.txt | awk '{print $1}' | grep -v -w '!SIL' | sort -u > $egsdir/wordlist.all

  cat $train_text | awk -v w=$egsdir/wordlist.all \
      'BEGIN{while((getline<w)>0) v[$1]=1;}
      {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}'|sed 's/ $//g' \
      | shuf --random-source=$train_text > $egsdir/train.txt.0

  cat $dev_text | awk -v w=$egsdir/wordlist.all \
      'BEGIN{while((getline<w)>0) v[$1]=1;}
      {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}'|sed 's/ $//g' \
        > $egsdir/dev.txt.0

  cat $egsdir/train.txt.0 $egsdir/wordlist.all | sed "s= =\n=g" | grep . | sort | uniq -c | sort -k1 -n -r | awk '{print $2,$1}' > $egsdir/unigramcounts.txt

  echo $bos 0 > $egsdir/wordlist.in
  echo $oos 1 >> $egsdir/wordlist.in
  cat $egsdir/unigramcounts.txt | head -n $num_words_in | awk '{print $1,1+NR}' >> $egsdir/wordlist.in

  echo $eos 0 > $egsdir/wordlist.out
  echo $oos 1 >> $egsdir/wordlist.out

  cat $egsdir/unigramcounts.txt | head -n $num_words_out | awk '{print $1,1+NR}' >> $egsdir/wordlist.out

  cat $egsdir/train.txt.0 | awk -v bos="$bos" -v eos="$eos" '{print bos,$0,eos}' > $egsdir/train.txt
  cat $egsdir/dev.txt.0   | awk -v bos="$bos" -v eos="$eos" '{print bos,$0,eos}' > $egsdir/dev.txt

  cat $egsdir/wordlist.all $egsdir/train.txt | awk '{for(i=1;i<=NF;i++) print $i}' | grep -v "<s>" \
     | awk -v w=$egsdir/wordlist.out \
      'BEGIN{while((getline<w)>0) {v[$1]=$2;}}
        {if(($1 in v) && (v[$1]!=1)){print(v[$1]);}else{printf("1\n");}}' | sort | uniq -c | awk '{print$2,$1}' | sort -k1 -n > $egsdir/uni_counts.txt

  cat $egsdir/uni_counts.txt | awk '{print NR-1, 1}' > $egsdir/uniform.txt
fi

num_words_in=`wc -l $egsdir/wordlist.in | awk '{print $1}'`
num_words_out=`wc -l $egsdir/wordlist.out | awk '{print $1}'`
num_words_total=`wc -l $egsdir/unigramcounts.txt  | awk '{print $1}'`

if [ $stage -le -3 ]; then
  echo Get Examples
  $cmd $egsdir/log/get-egs.train.log \
    rnnlm-get-egs $egsdir/train.txt $egsdir/wordlist.in $egsdir/wordlist.out ark,t:"$egsdir/train.egs" &
  $cmd $egsdir/log/get-egs.dev.log \
    rnnlm-get-egs $egsdir/dev.txt $egsdir/wordlist.in $egsdir/wordlist.out ark,t:"$egsdir/dev.egs"

  wait

  mkdir -p $egsdir/egs
  egs_str=
  for i in `seq 1 $num_archives`; do
    egs_str="$egs_str ark,t:$egsdir/egs/train.$i.egs"
  done

  nnet3-copy-egs ark:$egsdir/train.egs $egs_str

  $cmd $egsdir/log/create_train_subset_combine.log \
     nnet3-subset-egs --n=$num_train_frames_combine ark:$egsdir/train.egs \
     ark,t:$egsdir/train.subset.egs &                           

  cat $egsdir/dev.txt | shuf --random-source=$egsdir/dev.txt | head -n $num_frames_diagnostic > $egsdir/dev.diag.txt
  cat $egsdir/train.txt | shuf --random-source=$egsdir/train.txt | head -n $num_frames_diagnostic > $egsdir/train.diag.txt
  rnnlm-get-egs $egsdir/dev.diag.txt $egsdir/wordlist.in $egsdir/wordlist.out ark,t:"$egsdir/dev.subset.egs"
  rnnlm-get-egs $egsdir/train.diag.txt $egsdir/wordlist.in $egsdir/wordlist.out ark,t:"$egsdir/train_diagnostic.egs"
  wait
fi

oos_ratio=`cat $egsdir/dev.diag.txt | awk -v w=$egsdir/wordlist.out 'BEGIN{while((getline<w)>0) v[$1]=1;}
                                                         {for(i=2;i<=NF;i++){sum++; if(v[$i] != 1) oos++}} END{print oos/sum}'`

ppl_oos_penalty=`echo $num_words_out $num_words_total $oos_ratio | awk '{print ($2-$1)^$3}'`

echo dev oos ratio is $oos_ratio
echo dev oos penalty is $ppl_oos_penalty

unigram=$egsdir/uni_counts.txt

if [ $stage -le -2 ]; then
  echo Create nnet configs

  if [ "$type" == "rnn" ]; then
  cat > $outdir/config <<EOF
  LmNaturalGradientLinearComponent input-dim=$num_words_in output-dim=$hidden_dim max-change=2
  NaturalGradientAffineImportanceSamplingComponent learning-rate-factor=0.1 input-dim=$hidden_dim output-dim=$num_words_out max-change=2 num-samples-history=2000 unigram=$unigram

  input-node name=input dim=$hidden_dim
  component name=first_nonlin type=SigmoidComponent dim=$hidden_dim
  component name=hidden_affine type=NaturalGradientAffineComponent input-dim=$hidden_dim output-dim=$hidden_dim max-change=2

#Component nodes
  component-node name=first_nonlin component=first_nonlin  input=Sum(input, hidden_affine)
  component-node name=hidden_affine component=hidden_affine  input=IfDefined(Offset(first_nonlin, -1))

  output-node    name=output input=first_nonlin objective=linear
EOF
  fi

fi

if [ $stage -le 0 ]; then
  rnnlm-init --binary=false $outdir/config $outdir/0.mdl
fi

cat data/local/dict/lexicon.txt | awk '{print $1}' > $outdir/wordlist.all.1
cat $egsdir/wordlist.in $egsdir/wordlist.out | awk '{print $1}' > $outdir/wordlist.all.2
cat $outdir/wordlist.all.[12] | sort -u > $outdir/wordlist.all

cp $outdir/wordlist.all $outdir/wordlist.rnn
cp $egsdir/wordlist.in $outdir/rnn.wlist.in
cp $egsdir/wordlist.out $outdir/rnn.wlist.out

touch $outdir/unk.probs

mkdir -p $outdir/log/
if [ $stage -le $num_iters ]; then
  start=1
  learning_rate=$initial_learning_rate

  this_archive=0
  for n in `seq $start $num_iters`; do
    this_archive=$[$this_archive+1]

    [ $this_archive -gt $num_archives ] && this_archive=1

    echo for iter $n, training on archive $this_archive, learning rate = $learning_rate
    [ $n -ge $stage ] && (

        this_cmd=$cmd
        if [ "$use_gpu" == "yes" ]; then
          this_cmd=$cuda_cmd
        fi

        $this_cmd $outdir/log/train.$n.log rnnlm-train --adversarial-training-scale=$wang_scale --momentum=$momentum --verbose=2 --use-gpu=$use_gpu --binary=false \
        --max-param-change=$max_param_change "rnnlm-copy --learning-rate=$learning_rate $outdir/$[$n-1].mdl -|" \
        "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$n ark:$egsdir/egs/train.$this_archive.egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- | nnet3-copy-egs-add-samples --num-samples=$num_samples --srand=$n ark:- $unigram ark:- |" $outdir/$n.mdl

        if [ $n -gt 0 ]; then
          $cmd $outdir/log/progress.$n.log \
            rnnlm-show-progress --use-gpu=no $outdir/$[$n-1].mdl $outdir/$n.mdl \
            "ark:nnet3-merge-egs ark:$egsdir/train_diagnostic.egs ark:-|" '&&' \
            rnnlm-info $outdir/$n.mdl &
        fi

      echo -n "Iter $n "
      grep parse $outdir/log/train.$n.log | awk -F '-' '{print "Training PPL is " exp($NF)}'

    )

    learning_rate=`echo $learning_rate | awk -v d=$learning_rate_decline_factor '{printf("%f", $1/d)}'`
    if (( $(echo "$final_learning_rate > $learning_rate" |bc -l) )); then
      learning_rate=$final_learning_rate
    fi

    [ $n -ge $stage ] && (
      $decode_cmd $outdir/log/compute_prob_train.rnnlm.norm.$n.log \
        rnnlm-compute-prob --normalize-probs=true $outdir/$n.mdl "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$egsdir/train.subset.egs ark:- |" &
      $decode_cmd $outdir/log/compute_prob_valid.rnnlm.norm.$n.log \
        rnnlm-compute-prob --normalize-probs=true $outdir/$n.mdl "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$egsdir/dev.subset.egs ark:- |" &

      $decode_cmd $outdir/log/compute_prob_train.rnnlm.unnorm.$n.log \
        rnnlm-compute-prob --normalize-probs=false $outdir/$n.mdl "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$egsdir/train.subset.egs ark:- |" &
      $decode_cmd $outdir/log/compute_prob_valid.rnnlm.unnorm.$n.log \
        rnnlm-compute-prob --normalize-probs=false $outdir/$n.mdl "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$egsdir/dev.subset.egs ark:- |" 

      wait
      ppl=`grep Overall $outdir/log/compute_prob_train.rnnlm.norm.$n.log | grep like | awk '{print exp(-$8)}'`
      ppl2=`echo $ppl $ppl_oos_penalty | awk '{print $1 * $2}'`
      echo NORMALIZED TRAIN PPL on model $n.mdl is $ppl w/o OOS penalty, $ppl2 w OOS penalty

      ppl=`grep Overall $outdir/log/compute_prob_valid.rnnlm.norm.$n.log | grep like | awk '{print exp(-$8)}'`
      ppl2=`echo $ppl $ppl_oos_penalty | awk '{print $1 * $2}'`
      echo NORMALIZED DEV PPL on model $n.mdl is $ppl w/o OOS penalty, $ppl2 w OOS penalty

      ppl=`grep Overall $outdir/log/compute_prob_train.rnnlm.unnorm.$n.log | grep like | awk '{print exp(-$8)}'`
      ppl2=`echo $ppl $ppl_oos_penalty | awk '{print $1 * $2}'`
      echo UNNORMALIZED TRAIN PPL on model $n.mdl is $ppl w/o OOS penalty, $ppl2 w OOS penalty

      ppl=`grep Overall $outdir/log/compute_prob_valid.rnnlm.unnorm.$n.log | grep like | awk '{print exp(-$8)}'`
      ppl2=`echo $ppl $ppl_oos_penalty | awk '{print $1 * $2}'`
      echo UNNORMALIZED DEV PPL on model $n.mdl is $ppl w/o OOS penalty, $ppl2 w OOS penalty
    ) &

  done
  cp $outdir/$num_iters.mdl $outdir/rnnlm
fi
