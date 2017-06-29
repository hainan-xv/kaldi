#!/bin/bash

. cmd.sh
. path.sh

stage=2

mfccdir=mfcc

sfisher_speech=/export/a16/gkumar/corpora/LDC2010S01
sfisher_transcripts=/export/a16/gkumar/corpora/LDC2010T04
spanish_lexicon=/export/a16/gkumar/corpora/LDC96L16

if [ $stage -le 1 ]; then
  local/prepare_data.sh $sfisher_speech $sfisher_transcripts

#  steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir

fi

local/prepare_dict.sh $spanish_lexicon
utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
