#!/bin/bash

. cmd.sh
. path.sh

stage=9

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
  utils/fix_data_dir.sh data/test
  steps/decode.sh --nj 40 --cmd "$decode_cmd" \
    exp/mono/graph data/test exp/mono/decode_test
fi

if [ $stage -le 4 ]; then
  utils/subset_data_dir.sh data/train 100000 data/train_100k 
  utils/fix_data_dir.sh data/train_100k

  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
     data/train_100k data/lang exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd "$train_cmd" \
    3000 60000 data/train_100k data/lang exp/mono_ali exp/tri1

(  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph                              
  steps/decode.sh --nj 40 --cmd "$decode_cmd" \
    exp/tri1/graph data/test exp/tri1/decode_test ) &
fi

if [ $stage -le 5 ]; then
  utils/fix_data_dir.sh data/train
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
     data/train data/lang exp/tri1 exp/tri1_ali

  steps/train_deltas.sh --cmd "$train_cmd" \
      4000 100000 data/train data/lang exp/tri1_ali exp/tri2
  (
    utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
    steps/decode.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
     exp/tri2/graph data/test exp/tri2/decode_test
  )
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 100000 data/train data/lang exp/tri2_ali exp/tri3a

  (
    utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph
    steps/decode.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
     exp/tri3a/graph data/test exp/tri3a/decode_test
  )&
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train data/lang exp/tri3a exp/tri3a_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    4000 100000 data/train data/lang exp/tri3a_ali exp/tri4a

  (
    utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph_nosp
    steps/decode_fmllr.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
     exp/tri4a/graph_nosp data/test exp/tri4a/decode_test_nosp
  )&
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train data/lang exp/tri4a exp/tri4a_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 120000 data/train data/lang exp/tri4a_ali exp/tri5a

  (
    utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph_nosp
    steps/decode_fmllr.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
     exp/tri5a/graph_nosp data/test exp/tri5a/decode_test_nosp
  )
fi

# add silprobs
if [ $stage -le 9 ]; then
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang exp/tri5a
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict exp/tri5a/pron_counts_nowb.txt \
    exp/tri5a/sil_counts_nowb.txt \
    exp/tri5a/pron_bigram_counts_nowb.txt data/local/dict_sp

  utils/prepare_lang.sh data/local/dict_sp "<unk>" data/local/lang_sp data/lang_sp
  cp -rT data/lang_sp data/lang_sp_test
  cp data/lang_test/G.fst data/lang_sp_test

  utils/mkgraph.sh data/lang_sp_test exp/tri5a exp/tri5a/graph || exit 1

  steps/decode_fmllr.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri5a/graph data/test exp/tri5a/decode_test
  
fi

if [ $stage -le 10 ]; then

fi
