#!/bin/bash

. cmd.sh
. path.sh

sfisher_speech=/export/a16/gkumar/corpora/LDC2010S01
sfisher_transcripts=/export/a16/gkumar/corpora/LDC2010T04
spanish_lexicon=/export/a16/gkumar/corpora/LDC96L16

local/prepare_data.sh $sfisher_speech $sfisher_transcripts
