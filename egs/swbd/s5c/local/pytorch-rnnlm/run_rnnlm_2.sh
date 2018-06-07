#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Xiaohui Zhang

# This script trains LMs on the swbd LM-training data.

# rnnlm/train_rnnlm.sh: best iteration (out of 35) was 34, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 41.9 / 50.0.
# Train objf: -5.07 -4.43 -4.25 -4.17 -4.12 -4.07 -4.04 -4.01 -3.99 -3.98 -3.96 -3.94 -3.92 -3.90 -3.88 -3.87 -3.86 -3.85 -3.84 -3.83 -3.82 -3.81 -3.80 -3.79 -3.78 -3.78 -3.77 -3.77 -3.76 -3.75 -3.74 -3.73 -3.73 -3.72 -3.71
# Dev objf:   -10.32 -4.68 -4.43 -4.31 -4.24 -4.19 -4.15 -4.13 -4.10 -4.09 -4.05 -4.03 -4.02 -4.00 -3.99 -3.98 -3.98 -3.97 -3.96 -3.96 -3.95 -3.94 -3.94 -3.94 -3.93 -3.93 -3.93 -3.92 -3.92 -3.92 -3.92 -3.91 -3.91 -3.91 -3.91

# %WER 11.1 | 1831 21395 | 89.9 6.4 3.7 1.0 11.1 46.3 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped/score_13_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 9.9 | 1831 21395 | 91.0 5.8 3.2 0.9 9.9 43.2 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped_rnnlm_1e/score_11_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 9.9 | 1831 21395 | 91.0 5.8 3.2 0.9 9.9 42.9 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped_rnnlm_1e_nbest/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys

# %WER 15.9 | 4459 42989 | 85.7 9.7 4.6 1.6 15.9 51.6 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped/score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 14.4 | 4459 42989 | 87.0 8.7 4.3 1.5 14.4 49.4 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped_rnnlm_1e/score_11_0.0/eval2000_hires.ctm.filt.sys
# %WER 14.4 | 4459 42989 | 87.1 8.7 4.2 1.5 14.4 49.0 | exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp/decode_eval2000_sw1_fsh_fg_looped_rnnlm_1e_nbest/score_10_0.0/eval2000_hires.ctm.filt.sys

# Begin configuration section.

dir=data/pytorch_rnnlm
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=-10
train_stage=-10

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
run_backward_rnnlm=false

ac_model_dir=exp/nnet3/tdnn_lstm_1a_adversarial0.3_epochs12_ld5_sp
decode_dir_suffix=rnnlm_1e
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

text=data/train_nodev/text
fisher_text=data/local/lm/fisher/text1.gz
lexicon=data/local/dict_nosp/lexiconp.txt
mkdir -p $dir/config
set -e

mkdir -p $dir

cat data/rnnlm/text_nosp/swbd.txt data/rnnlm/text_nosp/fisher.txt | cut -d " " -f2- | shuf > $dir/all.txt

head -n 1000  $dir/all.txt                  > $dir/valid.txt
tail -n +1001 $dir/all.txt > $dir/train.txt
cat data/lang/words.txt | awk '{print $1}' > $dir/vocab.txt

if [ $stage -le 1 ]; then
  $cuda_cmd -l hostname=c* $dir/log.train_rnnlm CUDA_VISIBLE_DEVICES=\`free-gpu\` \&\& \
    python -u steps/pytorch-rnnlm-2/main.py --cuda --nhid 512 --dropout 0.2 --emsize 512 --data $dir

#         /home/tongfei/app/anaconda/bin/python -u steps/pytorch-rnnlm/rnnlm.py \
#         --CUDA --gpus \`free-gpu\` --MB_SIZE 64 \
#         --EMBED_SIZE 200 --HIDDEN_SIZE 200 --EPOCHS 10 \
#         --TRAIN $dir/train.txt --VALID $dir/valid.txt --VOCAB $dir/vocab.txt

fi

LM=sw1_fsh_fg # using the 4-gram const arpa file as old lm

if [ $stage -le 2 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in eval2000; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}_looped

    # Lattice rescoring
    steps/pytorch-rnnlm-2/lmrescore_nbest.sh \
      --cmd "$decode_cmd --mem 4G" --N 20 \
      --stage 6 \
      0.8 data/lang_$LM $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_pytorch_nbest_weighted_sqrt
  done
fi
