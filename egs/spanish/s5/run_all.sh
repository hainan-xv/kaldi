#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
train_nj=32
stage=0
# End configuration section
. ./utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

audio_data=/export/corpora/LDC/LDC98S74
transcript_data=/export/corpora/LDC/LDC98T29
eval_data=/export/corpora/LDC/LDC2001S91

sfisher_speech=/export/a16/gkumar/corpora/LDC2010S01
sfisher_transcripts=/export/a16/gkumar/corpora/LDC2010T04
spanish_lexicon=/export/a16/gkumar/corpora/LDC96L16

callhome_speech=/export/a16/gkumar/corpora/LDC96S35
callhome_transcripts=/export/a16/gkumar/corpora/LDC96T17

boost_sil=0.5
numLeavesTri1=1000
numGaussTri1=10000
numLeavesTri2=1000
numGaussTri2=20000
numLeavesTri3=6000
numGaussTri3=75000
numLeavesMLLT=6000
numGaussMLLT=75000
numLeavesSAT=6000
numGaussSAT=75000
unk="<unk>"

. ./cmd.sh
. ./path.sh

if [ $stage -le 0 ]; then
  # Eval dataset preparation
  
  # prepare_data.sh does not really care about the order or number of the 
  # corpus directories
  local/prepare_data_hub.sh \
    $eval_data/HUB4_1997NE/doc/h4ne97sp.sgm \
    $eval_data/HUB4_1997NE/h4ne_sp/h4ne97sp.sph data/eval_hub
  local/prepare_test_text_hub.pl \
    "$unk" data/eval_hub/text > data/eval_hub/text.clean
  mv data/eval_hub/text data/eval_hub/text.old
  mv data/eval_hub/text.clean data/eval_hub/text
  utils/fix_data_dir.sh data/eval_hub
fi

if [ $stage -le 1 ]; then
  ## Training dataset preparation
  local/prepare_data_hub.sh $audio_data $transcript_data data/train_hub
  local/prepare_training_text_hub.pl \
    "$unk" data/train_hub/text > data/train_hub/text.clean
  mv data/train_hub/text data/train_hub/text.old
  mv data/train_hub/text.clean data/train_hub/text
  utils/fix_data_dir.sh data/train_hub
fi

if [ $stage -le 2 ]; then
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

  local/train_lms_srilm_hub.sh  --oov-symbol "$unk"\
      --train-text data/train_hub/text data data/srilm_hub
  cp -R data/lang_nosp data/lang_test_hub
  utils/format_lm.sh \
    data/lang_nosp data/srilm_hub/lm.gz  data/local/dict_nosp/lexicon.txt data/lang_test_hub

fi

#if [ $stage -le 2 ]; then
#  # Graphemic lexicon
#  mkdir -p data/local_hub
#  local/prepare_lexicon_hub.sh data/train_hub/text data/local_hub
#fi
#exit

if [ $stage -le 3 ]; then
  # Training set features
	steps/make_mfcc.sh --cmd "$train_cmd" --nj $train_nj --mfcc_config conf/mfcc.conf.hub data/train_hub exp/make_mfcc/train_hub mfcc
  utils/fix_data_dir.sh data/train_hub
  steps/compute_cmvn_stats.sh data/train_hub exp/make_mfcc/train_hub mfcc
  utils/fix_data_dir.sh data/train_hub
fi

if [ $stage -le 4 ]; then
  # Eval dataset features
	steps/make_mfcc.sh --cmd "$decode_cmd" --nj 16 --mfcc_config conf/mfcc.conf.hub data/eval_hub exp/make_mfcc/eval_hub mfcc
  utils/fix_data_dir.sh data/eval_hub
  steps/compute_cmvn_stats.sh data/eval_hub exp/make_mfcc/eval_hub mfcc
  utils/fix_data_dir.sh data/eval_hub
fi

if [ $stage -le 5 ]; then
  for i in train_callhome test_callhome train_fisher test_fisher; do
    steps/make_mfcc.sh --nj 80 --cmd "$train_cmd" data/$i exp/make_mfcc/$i mfcc
    steps/compute_cmvn_stats.sh data/$i exp/make_mfcc/$i mfcc
  done
  utils/combine_data.sh data/train data/train_callhome data/train_fisher data/train_hub
  utils/combine_data.sh data/test data/test_callhome data/test_fisher data/eval_hub
fi

# a monophone system
if [ $stage -le 6 ]; then
  utils/subset_data_dir.sh data/train 10000 data/train_10k 
  utils/fix_data_dir.sh data/train_10k

  utils/data/remove_dup_utts.sh 100 data/train_10k data/train_10k_nodup

  steps/train_mono.sh --nj 40 --cmd "$train_cmd" data/train_10k_nodup data/lang_nosp exp/mono
(  utils/mkgraph.sh data/lang_nosp_test exp/mono exp/mono/graph

  steps/decode.sh --nj 40 --cmd "$decode_cmd" \
    exp/mono/graph data/test exp/mono/decode_test) &
fi

if [ $stage -le 7 ]; then
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

if [ $stage -le 7 ]; then
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

if [ $stage -le 7 ]; then
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
