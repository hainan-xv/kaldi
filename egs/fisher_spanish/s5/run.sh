#!/bin/bash

. cmd.sh
. path.sh

stage=3

mfccdir=mfcc

sfisher_speech=/export/a16/gkumar/corpora/LDC2010S01
sfisher_transcripts=/export/a16/gkumar/corpora/LDC2010T04
spanish_lexicon=/export/a16/gkumar/corpora/LDC96L16

if [ $stage -le 1 ]; then
  local/prepare_data.sh $sfisher_speech $sfisher_transcripts

  utils/fix_data_dir.sh data/train
  utils/fix_data_dir.sh data/test

  local/prepare_dict.sh $spanish_lexicon
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  local/prepare_lm.sh
fi

if [ $stage -le 2 ]; then
  for i in train test; do
    steps/make_mfcc_pitch.sh --nj 80 --cmd "$train_cmd" data/$i exp/make_mfcc/$i $mfccdir
    steps/compute_cmvn_stats.sh data/$i exp/make_feat/$i $mfccdir
  done
fi

# a monophone system
if [ $stage -le 3 ]; then
  utils/subset_data_dir.sh data/train 10000 data/train_10k 
  utils/fix_data_dir.sh data/train_10k

  utils/data/remove_dup_utts.sh 100 data/train_10k data/train_10k_nodup

  steps/train_mono.sh --nj 40 --cmd "$train_cmd" data/train_10k_nodup data/lang exp/mono
  utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph                              
  steps/decode.sh --nj 100 --cmd "$decode_cmd" \
    exp/mono/graph data/test exp/mono/decode_test
fi

if [ $stage -le 3 ]; then
  utils/subset_data_dir.sh data/train 100000 data/train_100k 
  utils/fix_data_dir.sh data/train_100k

  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
     data/train_100k data/lang exp/mono exp/mono0a_ali || exit 1;

  steps/train_deltas --nj 40 --cmd "$train_cmd" \
    4000 100000 data/train_100k_nodup data/lang exp/tri1
  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph                              
  steps/decode.sh --nj 100 --cmd "$decode_cmd" \
    exp/tri1/graph data/test exp/tri1/decode_test
fi


