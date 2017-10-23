#!/bin/bash

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

mic=ihm
ngram_order=4 # this option when used, the rescoring binary makes an approximation

rnndir=model/h512.mb64.chunk6.ce
rnndir=my_cued_model_200/

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

weight=0.5 # the weight of the RNNLM in rescoring

#for decode_set in eval; do
for decode_set in dev eval; do
  basedir=exp/$mic/nnet3/tdnn_sp/
  decode_dir=${basedir}/decode_${decode_set}

  # Lattice rescoring
#  ./cuedrnnlm_lat_rescore_pruned.sh \
#  ./cuedrnnlm_lat_rescore.sh \
  ./cuedrnnlm_lat_rescore_pruned.sh \
    --cmd "$decode_cmd -l hostname=b*" \
    --weight $weight --max-ngram-order $ngram_order \
    data/lang_$LM $rnndir \
    data/$mic/${decode_set}_hires ${decode_dir} \
    ${decode_dir}.cued-lstm.lat.${ngram_order}gram.$weight.new_model.pruned.200 &

#  rnnlm/lmrescore_rnnlm_lat \
#    --cmd "$decode_cmd -l hostname=b*" \
#    --weight $weight --max-ngram-order $ngram_order \
#    data/lang_$LM $rnndir \
#    data/${decode_set}_hires ${decode_dir} \
#    ${decode_dir}.kaldirnnlm.lat.${ngram_order}gram.$weight
done

wait
