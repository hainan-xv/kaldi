#!/bin/bash

. cmd.sh
. path.sh

set -e
stage=1

mfccdir=mfcc

sfisher_speech=/export/a16/gkumar/corpora/LDC2010S01
sfisher_transcripts=/export/a16/gkumar/corpora/LDC2010T04
spanish_lexicon=/export/a16/gkumar/corpora/LDC96L16

callhome_speech=/export/a16/gkumar/corpora/LDC96S35
callhome_transcripts=/export/a16/gkumar/corpora/LDC96T17

. parse_options.sh || exit 1;

if [ $stage -le 1 ]; then
  local/prepare_data.sh $sfisher_speech $sfisher_transcripts
  local/prepare_callhome_data.sh $callhome_speech $callhome_transcripts
  local/prepare_giga.sh

  utils/fix_data_dir.sh data/train_fisher
  utils/fix_data_dir.sh data/test_fisher


  local/prepare_dict.sh $spanish_lexicon
  mkdir -p data/local/dict_nosp/

  mv data/local/dict/* data/local/dict_nosp/
  utils/prepare_lang.sh data/local/dict_nosp "<unk>" data/local/lang_nosp data/lang_nosp

  local/prepare_lm.sh
  local/prepare_lm_giga.sh
fi

if [ $stage -le 2 ]; then
  for i in train_callhome test_callhome train_fisher test_fisher; do
    steps/make_mfcc.sh --nj 80 --cmd "$train_cmd" data/$i exp/make_mfcc/$i $mfccdir
    steps/compute_cmvn_stats.sh data/$i exp/make_feat/$i $mfccdir
  done
  utils/combine_data.sh data/train data/train_callhome data/train_fisher
  utils/combine_data.sh data/test data/test_callhome data/test_fisher
fi

# a monophone system
if [ $stage -le 3 ]; then
  utils/subset_data_dir.sh data/train 10000 data/train_10k 
  utils/fix_data_dir.sh data/train_10k

  utils/data/remove_dup_utts.sh 100 data/train_10k data/train_10k_nodup

  steps/train_mono.sh --nj 40 --cmd "$train_cmd" data/train_10k_nodup data/lang_nosp exp/mono
(  utils/mkgraph.sh data/lang_nosp_test exp/mono exp/mono/graph

  steps/decode.sh --nj 40 --cmd "$decode_cmd" \
    exp/mono/graph data/test exp/mono/decode_test) &
fi

if [ $stage -le 4 ]; then
  utils/subset_data_dir.sh data/train 100000 data/train_100k 
  utils/fix_data_dir.sh data/train_100k

  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
     data/train_100k data/lang_nosp exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd "$train_cmd" \
    3000 60000 data/train_100k data/lang_nosp exp/mono_ali exp/tri1

(  utils/mkgraph.sh data/lang_nosp_test exp/tri1 exp/tri1/graph                              
  steps/decode.sh --nj 40 --cmd "$decode_cmd" \
    exp/tri1/graph data/test exp/tri1/decode_test ) &
fi

if [ $stage -le 5 ]; then
  utils/fix_data_dir.sh data/train
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
     data/train data/lang_nosp exp/tri1 exp/tri1_ali

  steps/train_deltas.sh --cmd "$train_cmd" \
      4000 100000 data/train data/lang_nosp exp/tri1_ali exp/tri2
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri2 exp/tri2/graph
    steps/decode.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
     exp/tri2/graph data/test exp/tri2/decode_test
  ) &
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri2 exp/tri2_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 100000 data/train data/lang_nosp exp/tri2_ali exp/tri3a

  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri3a exp/tri3a/graph
    steps/decode.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
     exp/tri3a/graph data/test exp/tri3a/decode_test
  )&
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri3a exp/tri3a_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    4000 200000 data/train data/lang_nosp exp/tri3a_ali exp/tri4a

  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri4a exp/tri4a/graph_nosp
    steps/decode_fmllr.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
     exp/tri4a/graph_nosp data/test exp/tri4a/decode_test_nosp
  )&
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri4a exp/tri4a_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 250000 data/train data/lang_nosp exp/tri4a_ali exp/tri5a

  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri5a exp/tri5a/graph_nosp
    steps/decode_fmllr.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
     exp/tri5a/graph_nosp data/test exp/tri5a/decode_test_nosp
  )
fi

# add silprobs
if [ $stage -le 9 ]; then
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_nosp exp/tri5a
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp exp/tri5a/pron_counts_nowb.txt \
    exp/tri5a/sil_counts_nowb.txt \
    exp/tri5a/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  cp -rT data/lang data/lang_test
  cp data/lang_nosp_test/G.fst data/lang_test

(  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1

  steps/decode_fmllr.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri5a/graph data/test exp/tri5a/decode_test
   ) &  
fi

if [ $stage -le 10 ]; then
  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
    data/train data/lang exp/tri5a exp/tri5a_ali || exit 1

fi
