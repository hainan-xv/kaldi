#!/bin/bash

. path.sh
. cmd.sh

mkdir -p data/pytorch-lm

cat data/train/text | cut -d " " -f2- | shuf > data/pytorch-lm/all.txt

head -n 1000 data/pytorch-lm/all.txt > data/pytorch-lm/valid.txt
tail -n +1001 data/pytorch-lm/all.txt > data/pytorch-lm/train.txt

export PATH=/home/tongfei/app/anaconda/bin:$PATH

$cuda_cmd -l hostname=c* data/pytorch-lm/log.train_rnnlm CUDA_VISIBLE_DEVICES=\`free-gpu\` \&\& /home/tongfei/app/anaconda/bin/python -u local/pytorch-rnnlm/main.py --cuda --nhid 512 --dropout 0.4
