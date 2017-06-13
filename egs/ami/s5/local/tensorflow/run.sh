#!/bin/bash
mic=ihm
ngram_order=4
model_type=small
dir=$PWD/data/tensorflow
stage=3

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

mkdir -p $dir

if [ $stage -le 1 ]; then
# num-words is 10000 - 3 (bos, eos and <oos>)
  cat data/$mic/train/text | awk '{for(i=2;i<=NF;i++)print $i}' | sort | uniq -c | sort -k1nr | head -n 9997 | awk '{print $2}' > $dir/wordlist

  for i in train dev eval; do
    cat data/$mic/$i/text | awk -v w=$dir/wordlist 'BEGIN{while((getline<w)>0)d[$1]=1}{for(i=2;i<=NF;i++){if(d[$i]==1){s=$i}else{s="<oos>"} printf("%s ",s)} print""}' | sed "s=^= <s> =g" | sed "s=$= </s>=" > $dir/$i.txt
  done
fi

if [ $stage -le 2 ]; then
  python local/tensorflow/rnnlm.py --data_path=$dir --model=small --save_path=$dir/rnnlm --wordlist_save_path=$dir/wordlist.rnn
#python local/tensorflow/rnnlm.py --data_path=$dir --model=medium --save_path=$dir/model.medium
#python local/tensorflow/rnnlm.py --data_path=$dir --model=large --save_path=$dir/model.large
fi

touch $dir/unk.probs

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

if [ $stage -le 3 ]; then
  for decode_set in dev eval; do
    basedir=exp/$mic/nnet3/tdnn_sp/
    decode_dir=${basedir}/decode_${decode_set}

    # Lattice rescoring
    steps/lmrescore_rnnlm_lat.sh \
      --cmd "$decode_cmd --mem 16G" \
      --rnnlm-ver tensorflow  --weight 0.5 --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/$mic/${decode_set}_hires ${decode_dir} \
      ${decode_dir}.tfrnnlm.lat.${ngram_order}gram

  done
fi
