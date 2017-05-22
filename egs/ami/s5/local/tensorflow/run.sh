#!/bin/bash

data_type=sdm1
model_type=small

dir=data/tensorflow/
mkdir -p $dir

cat data/$data_type/train/text | awk '{for(i=2;i<=NF;i++)print $i}' | sort | uniq -c | sort -k1nr | head -n 9998 | awk '{print $2}' > $dir/wordlist

for i in train dev eval; do
  cat data/$data_type/$i/text | awk -v w=$dir/wordlist 'BEGIN{while((getline<w)>0)d[$1]=1}{for(i=2;i<=NF;i++){if(d[$i]==1){s=$i}else{s="<oos>"} printf("%s ",s)} print""}' | sed "s=^= =g" > $dir/$i.txt
done

python local/tensorflow/ptb_word_lm.py --data_path=$dir --model=$model_type --save_path=$dir/rnnlm.mdl
