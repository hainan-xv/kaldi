#!/bin/bash

mic=sdm1
n=50
ngram_order=4
rnndir=data/nnet3_rnnlm_200_256_0
id=rnn

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

local/nnet3-rnnlm/run-rnnlm-train.sh --use-gpu yes --stage -20 --num-iters 160

[ ! -f $rnndir/rnnlm ] && echo "Can't find RNNLM model" && exit 1;

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

for decode_set in dev eval; do
(  dir=exp/$mic/nnet3/tdnn_sp/
  decode_dir=${dir}/decode_${decode_set}

  steps/lmrescore_rnnlm_lat.sh \
    --cmd "$decode_cmd --mem 16G" \
    --rnnlm-ver nnet3rnnlm  --weight 0.5 --max-ngram-order $ngram_order \
    data/lang_$LM $rnndir \
    data/$mic/${decode_set}_hires ${decode_dir} \
    ${decode_dir}.rnnlm.lat.${ngram_order}gram
) &
done

wait
