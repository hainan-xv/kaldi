#!/bin/bash

mic=ihm
n=50
ngram_order=4
rnndir=data/nnet3_rnnlm
id=rnn

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

local/nnet3-rnnlm/run-rnnlm-train.sh --use-gpu yes --stage 41

[ ! -f $rnndir/rnnlm ] && echo "Can't find RNNLM model" && exit 1;

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

for decode_set in dev eval; do
(  dir=exp/$mic/nnet3_cleaned/tdnn_sp/
  decode_dir=${dir}/decode_${decode_set}

  steps/lmrescore_rnnlm_lat.sh \
    --cmd "$decode_cmd --mem 16G" \
    --rnnlm-ver nnet3rnnlm  --weight 0.5 --max-ngram-order $ngram_order \
    data/lang_$LM data/nnet3_rnnlm \
    data/$mic/${decode_set}_hires ${decode_dir} \
    ${decode_dir}.rnnlm.lat.${ngram_order}gram
) &
done

wait
