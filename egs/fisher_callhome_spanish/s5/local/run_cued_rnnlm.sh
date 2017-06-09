#!/bin/bash

crit=vr
n=50
ngram_order=4

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

mkdir -p data/cued_rnn_$crit

#cat data/local/lm/text.no_oov | awk '{for(i=2;i<=NF;i++)printf("%s ",$i); print""}' | grep . > data/cued_rnn_$crit/all.txt

#local/train_cued_rnnlms.sh --crit $crit --train-text data/local/lm/text.no_oov data/cued_rnn_$crit

LM=test

for decode_set in dev test; do
  dir=exp/tri5a
  decode_dir=${dir}/decode_${decode_set}

  # N-best rescoring
  steps/rnnlmrescore.sh \
    --rnnlm-ver cuedrnnlm \
    --N $n --cmd "$decode_cmd --mem 16G" --inv-acwt 10 0.5 \
    data/lang_$LM data/cued_rnn_$crit \
    data/$decode_set ${decode_dir} \
    ${decode_dir}.rnnlm.$crit.cued.$n-best &

  # Lattice rescoring
  steps/lmrescore_rnnlm_lat.sh \
    --cmd "$decode_cmd --mem 16G" \
    --rnnlm-ver cuedrnnlm  --weight 0.5 --max-ngram-order $ngram_order \
    data/lang_$LM data/$mic/cued_rnn_$crit \
    data/${decode_set}_hires ${decode_dir} \
    ${decode_dir}.rnnlm.$crit.cued.lat.${ngram_order}gram &

done
