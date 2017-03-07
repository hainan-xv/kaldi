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
num_iters=40

num_train_frames_combine=10000 # # train frames for the above.                  
num_frames_diagnostic=2000 # number of frames for "compute_prob" jobs  
num_archives=4

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
minibatch_size=1:128

hidden_dim=150
initial_learning_rate=0.1
final_learning_rate=0.01
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

wang_interval=1
wang_scale=0.2

type=rnn  # or lstm

id=

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

outdir=rnnlm_nnet3_${hidden_dim}_${initial_learning_rate}_wang_update_${wang_interval}_${wang_scale}
#outdir=rnnlm_nnet3_${hidden_dim}_${initial_learning_rate}
srcdir=data/local/dict

set -e

mkdir -p $outdir

if [ $stage -le -4 ]; then
  echo Data Preparation
  cat $srcdir/lexicon.txt | awk '{print $1}' | grep -v -w '!SIL' | sort -u > $outdir/wordlist.all

  cat $train_text | awk -v w=$outdir/wordlist.all \
      'BEGIN{while((getline<w)>0) v[$1]=1;}
      {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}'|sed 's/ $//g' \
      | shuf --random-source=$train_text > $outdir/train.txt.0

  cat $dev_text | awk -v w=$outdir/wordlist.all \
      'BEGIN{while((getline<w)>0) v[$1]=1;}
      {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}'|sed 's/ $//g' \
        > $outdir/dev.txt.0

  cat $outdir/train.txt.0 $outdir/wordlist.all | sed "s= =\n=g" | grep . | sort | uniq -c | sort -k1 -n -r | awk '{print $2,$1}' > $outdir/unigramcounts.txt

  echo $bos 0 > $outdir/wordlist.in
  echo $oos 1 >> $outdir/wordlist.in
  cat $outdir/unigramcounts.txt | head -n $num_words_in | awk '{print $1,1+NR}' >> $outdir/wordlist.in

  echo $eos 0 > $outdir/wordlist.out
  echo $oos 1 >> $outdir/wordlist.out

  cat $outdir/unigramcounts.txt | head -n $num_words_out | awk '{print $1,1+NR}' >> $outdir/wordlist.out

  cat $outdir/train.txt.0 | awk -v bos="$bos" -v eos="$eos" '{print bos,$0,eos}' > $outdir/train.txt
  cat $outdir/dev.txt.0   | awk -v bos="$bos" -v eos="$eos" '{print bos,$0,eos}' > $outdir/dev.txt

  cat $outdir/wordlist.all $outdir/train.txt | awk '{for(i=1;i<=NF;i++) print $i}' | grep -v "<s>" \
     | awk -v w=$outdir/wordlist.out \
      'BEGIN{while((getline<w)>0) {v[$1]=$2;}}
        {if(($1 in v) && (v[$1]!=1)){print(v[$1]);}else{printf("1\n");}}' | sort | uniq -c | awk '{print$2,$1}' | sort -k1 -n > $outdir/uni_counts.txt

  cat $outdir/uni_counts.txt | awk '{print NR-1, 1}' > $outdir/uniform.txt
fi

num_words_in=`wc -l $outdir/wordlist.in | awk '{print $1}'`
num_words_out=`wc -l $outdir/wordlist.out | awk '{print $1}'`
num_words_total=`wc -l $outdir/unigramcounts.txt  | awk '{print $1}'`

if [ $stage -le -3 ]; then
  echo Get Examples
  $cmd $outdir/log/get-egs.train.txt \
    rnnlm-get-egs $outdir/train.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"$outdir/train.egs" &
  $cmd $outdir/log/get-egs.dev.txt \
    rnnlm-get-egs $outdir/dev.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"$outdir/dev.egs"

  wait

  mkdir -p $outdir/egs
  egs_str=
  for i in `seq 1 $num_archives`; do
    egs_str="$egs_str ark:$outdir/egs/train.$i.egs"
  done

  nnet3-copy-egs ark:$outdir/train.egs $egs_str

  $cmd $outdir/log/create_train_subset_combine.log \
     nnet3-subset-egs --n=$num_train_frames_combine ark:$outdir/train.egs \
     ark,t:$outdir/train.subset.egs &                           

  cat $outdir/dev.txt | shuf --random-source=$outdir/dev.txt | head -n $num_frames_diagnostic > $outdir/dev.diag.txt
  cat $outdir/train.txt | shuf --random-source=$outdir/train.txt | head -n $num_frames_diagnostic > $outdir/train.diag.txt
  rnnlm-get-egs $outdir/dev.diag.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"$outdir/dev.subset.egs"
  rnnlm-get-egs $outdir/train.diag.txt $outdir/wordlist.in $outdir/wordlist.out ark,t:"$outdir/train_diagnostic.egs"

  wait
fi

oos_ratio=`cat $outdir/dev.diag.txt | awk -v w=$outdir/wordlist.out 'BEGIN{while((getline<w)>0) v[$1]=1;}
                                                         {for(i=2;i<=NF;i++){sum++; if(v[$i] != 1) oos++}} END{print oos/sum}'`

ppl_oos_penalty=`echo $num_words_out $num_words_total $oos_ratio | awk '{print ($2-$1)^$3}'`

echo dev oos ratio is $oos_ratio
echo dev oos penalty is $ppl_oos_penalty

if [ $stage -le -2 ]; then

#  echo Create nnet configs
#  if [ "$type" == "dnn" ]; then
#  cat > $outdir/config <<EOF
#
#  input-node name=input dim=$num_words_in
#  component name=first_affine type=AffineComponent input-dim=$[$num_words_in] output-dim=$hidden_dim max-change=5
#  component name=first_nonlin type=SigmoidComponent dim=$hidden_dim
##  component name=first_renorm type=NormalizeComponent dim=$hidden_dim target-rms=1.0
#  component name=final_affine type=AffineComponent input-dim=$hidden_dim output-dim=$num_words_out max-change=5
#  component name=final_log_softmax type=LogSoftmaxComponent dim=$num_words_out
##Component nodes
#  component-node name=first_affine component=first_affine  input=input
#  component-node name=first_nonlin component=first_nonlin  input=first_affine
##  component-node name=first_renorm component=first_renorm  input=first_nonlin
#  component-node name=final_affine component=final_affine  input=first_nonlin
#  component-node name=final_log_softmax component=final_log_softmax input=final_affine
#  output-node    name=output input=final_log_softmax objective=linear
#EOF
#  fi
#
#  if [ "$type" == "rnn" ]; then
#  cat > $outdir/config <<EOF
#
#  input-node name=input dim=$num_words_in
#  component name=first_affine type=AffineComponent input-dim=$[$num_words_in+$hidden_dim] output-dim=$hidden_dim max-change=5
#  component name=first_nonlin type=SigmoidComponent dim=$hidden_dim
#  component name=first_renorm type=NormalizeComponent dim=$hidden_dim target-rms=1.0
#  component name=final_affine type=AffineComponent input-dim=$hidden_dim output-dim=$num_words_in max-change=5
#  component name=final_log_softmax type=LogSoftmaxComponent dim=$num_words_in
##Component nodes
#  component-node name=first_affine component=first_affine  input=Append(input, IfDefined(Offset(first_renorm, -1)))
#  component-node name=first_nonlin component=first_nonlin  input=first_affine
#  component-node name=first_renorm component=first_renorm  input=first_nonlin
#  component-node name=final_affine component=final_affine  input=first_nonlin
#  component-node name=final_log_softmax component=final_log_softmax input=final_affine
#  output-node    name=output input=final_log_softmax objective=linear
#EOF
#  fi

  if [ "$type" == "rnn" ]; then
  cat > $outdir/config <<EOF
  input-node name=input dim=$num_words_in
#  component name=first_affine type=LinearComponent input-dim=$[$num_words_in] output-dim=$hidden_dim max-change=5
  component name=first_affine type=NaturalGradientLinearComponent input-dim=$[$num_words_in] output-dim=$hidden_dim max-change=5
  component name=recur_affine type=NaturalGradientAffineComponent input-dim=$[$hidden_dim] output-dim=$hidden_dim max-change=5
  component name=first_nonlin type=SigmoidComponent dim=$hidden_dim
#  component name=first_renorm type=NormalizeComponent dim=$hidden_dim target-rms=1.0
  component name=final_affine type=NaturalGradientAffineComponent input-dim=$hidden_dim output-dim=$num_words_out max-change=5
  component name=final_log_softmax type=LogSoftmaxComponent dim=$num_words_out

#Component nodes
  component-node name=first_affine component=first_affine  input=input
  component-node name=recur_affine component=recur_affine  input=IfDefined(Offset(first_nonlin, -1))
  component-node name=first_nonlin component=first_nonlin  input=Sum(first_affine, recur_affine)
#  component-node name=first_renorm component=first_renorm  input=first_nonlin
  component-node name=final_affine component=final_affine  input=first_nonlin
  component-node name=final_log_softmax component=final_log_softmax input=final_affine
  output-node    name=output input=final_log_softmax objective=linear

#  input-node name=input dim=$num_words_in
#  component name=first_affine type=NaturalGradientLinearComponent input-dim=$[$num_words_in] output-dim=$hidden_dim max-change=5
#  component name=recur_affine type=NaturalGradientAffineComponent input-dim=$[$hidden_dim] output-dim=$hidden_dim max-change=5
#  component name=first_nonlin type=RectifiedLinearComponent dim=$hidden_dim
#  component name=first_renorm type=NormalizeComponent dim=$hidden_dim
#  component name=final_affine type=NaturalGradientAffineComponent input-dim=$hidden_dim output-dim=$num_words_out max-change=5
#  component name=final_log_softmax type=LogSoftmaxComponent dim=$num_words_out
#
##Component nodes
#  component-node name=first_affine component=first_affine  input=input
#  component-node name=recur_affine component=recur_affine  input=IfDefined(Offset(first_nonlin, -1))
#  component-node name=first_nonlin component=first_nonlin  input=Sum(first_affine, recur_affine)
#  component-node name=first_renorm component=first_renorm  input=first_nonlin
#  component-node name=final_affine component=final_affine  input=first_renorm
#  component-node name=final_log_softmax component=final_log_softmax input=final_affine
#  output-node    name=output input=final_log_softmax objective=linear

EOF
#  input-node name=input dim=$num_words_in
##  component name=first_affine type=LinearComponent input-dim=$[$num_words_in] output-dim=$hidden_dim max-change=5
#  component name=first_affine type=NaturalGradientLinearComponent input-dim=$[$num_words_in] output-dim=$hidden_dim max-change=5
#  component name=recur_affine type=NaturalGradientAffineComponent input-dim=$[$hidden_dim] output-dim=$hidden_dim max-change=5
#  component name=first_nonlin type=SigmoidComponent dim=$hidden_dim
##  component name=first_renorm type=NormalizeComponent dim=$hidden_dim target-rms=1.0
#  component name=final_affine type=NaturalGradientAffineComponent input-dim=$hidden_dim output-dim=$num_words_out max-change=5
#  component name=final_log_softmax type=LogSoftmaxComponent dim=$num_words_out
#
##Component nodes
#  component-node name=first_affine component=first_affine  input=input
#  component-node name=recur_affine component=recur_affine  input=IfDefined(Offset(first_nonlin, -1))
#  component-node name=first_nonlin component=first_nonlin  input=Sum(first_affine, recur_affine)
##  component-node name=first_renorm component=first_renorm  input=first_nonlin
#  component-node name=final_affine component=final_affine  input=first_nonlin
#  component-node name=final_log_softmax component=final_log_softmax input=final_affine
#  output-node    name=output input=final_log_softmax objective=linear
  fi
fi

if [ $stage -le 0 ]; then
  nnet3-init --binary=false $outdir/config $outdir/0.mdl
fi

cat data/local/dict/lexicon.txt | awk '{print $1}' > $outdir/wordlist.all.1
cat $outdir/wordlist.in $outdir/wordlist.out | awk '{print $1}' > $outdir/wordlist.all.2
cat $outdir/wordlist.all.[12] | sort -u > $outdir/wordlist.all

cp $outdir/wordlist.all $outdir/wordlist.rnn
touch $outdir/unk.probs
#rm $outdir/wordlist.all.[12]


mkdir -p $outdir/log/
if [ $stage -le $num_iters ]; then
  start=1
#  if [ $stage -gt 1 ]; then
#    start=$stage
#  fi
  learning_rate=$initial_learning_rate

  this_archive=0
  for n in `seq $start $num_iters`; do
    this_archive=$[$this_archive+1]

    [ $this_archive -gt $num_archives ] && this_archive=1

    echo for iter $n, training on archive $this_archive, learning rate = $learning_rate
    [ $n -ge $stage ] && (

        unigram=$outdir/uni_counts.txt

        if [ $n -lt -1 ]; then
          unigram=$outdir/uniform.txt
        fi

#        $cuda_cmd $outdir/log/train.rnnlm.$n.log nnet3-train --use-gpu=$use_gpu --binary=false \

        $cuda_cmd $outdir/log/train.rnnlm.$n.log nnet3-train --use-gpu=$use_gpu --binary=false \
        --adversarial-training-interval=$wang_interval --adversarial-training-scale=$wang_scale \
        --max-param-change=$max_param_change "nnet3-copy --learning-rate=$learning_rate $outdir/$[$n-1].mdl -|" \
        "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$n ark:$outdir/egs/train.$this_archive.egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" $outdir/$n.mdl 

        (
        if [ $n -gt 0 ]; then
          $cmd $outdir/log/progress.$n.log \
            nnet3-show-progress --use-gpu=no $outdir/$[$n-1].mdl $outdir/$n.mdl \
            "ark:nnet3-merge-egs ark:$outdir/train_diagnostic.egs ark:-|" '&&' \
            nnet3-info $outdir/$n.mdl &
        fi
        )

#      t=`grep "^# Accounting" $outdir/log/train.rnnlm.$n.log | sed "s/=/ /g" | awk '{print $4}'`
#      w=`wc -w $outdir/splitted-text/train.$this_archive.txt | awk '{print $1}'`
#      speed=`echo $w $t | awk '{print $1/$2}'`
#      echo Processing speed: $speed words per second \($w words in $t seconds\)

      grep parse $outdir/log/train.rnnlm.$n.log | awk -F '-' '{print "Training PPL is " exp($NF)}'

    )

    learning_rate=`echo $learning_rate | awk -v d=$learning_rate_decline_factor '{printf("%f", $1/d)}'`
    if (( $(echo "$final_learning_rate > $learning_rate" |bc -l) )); then
      learning_rate=$final_learning_rate
    fi

    [ $n -ge $stage ] && (
      $decode_cmd $outdir/log/compute_prob_train.rnnlm.$n.log \
        nnet3-compute-prob $outdir/$n.mdl ark:$outdir/train.subset.egs &
      $decode_cmd $outdir/log/compute_prob_valid.rnnlm.$n.log \
        nnet3-compute-prob $outdir/$n.mdl ark:$outdir/dev.subset.egs 

      wait
      ppl=`grep Overall $outdir/log/compute_prob_train.rnnlm.$n.log | grep like | awk '{print exp(-$8)}'`
      ppl2=`echo $ppl $ppl_oos_penalty | awk '{print $1 * $2}'`
      echo TRAIN PPL on model $n.mdl is $ppl w/o OOS penalty, $ppl2 w OOS penalty

      ppl=`grep Overall $outdir/log/compute_prob_valid.rnnlm.$n.log | grep like | awk '{print exp(-$8)}'`
      ppl2=`echo $ppl $ppl_oos_penalty | awk '{print $1 * $2}'`
      echo DEV PPL on model $n.mdl is $ppl w/o OOS penalty, $ppl2 w OOS penalty
    ) &

  done
  cp $outdir/$num_iters.mdl $outdir/rnnlm
fi

#./local/rnnlm/run-rescoring.sh --rnndir $outdir/ --id $id
