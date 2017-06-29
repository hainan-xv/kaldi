#!/bin/bash

. cmd.sh
. path.sh

mfccdir=mfcc

sfisher_speech=/export/a16/gkumar/corpora/LDC2010S01
sfisher_transcripts=/export/a16/gkumar/corpora/LDC2010T04
spanish_lexicon=/export/a16/gkumar/corpora/LDC96L16

local/prepare_data.sh $sfisher_speech $sfisher_transcripts

utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt

steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir
