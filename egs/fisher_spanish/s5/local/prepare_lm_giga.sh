#!/bin/bash

tmpdir=data/local/gigatmp
stage=1

if [ $stage -le 2 ]; then
#zcat $tmpdir/cleantext.txt.gz | 
  ngram-count -text $tmpdir/cleantext.head.txt -order 3 -limit-vocab data/local/lm/wlist -kndiscount -lm data/local/lm/giga.arpa.gz
fi

if [ $stage -le 3 ]; then
#  ngram -lm data/local/lm/giga.arpa.gz -ppl data/local/lm/dev

  oov_symbol="<unk>"
#  for a in 03 04 05 06; do
  for a in 03 04 05 06 07 08 09 10; do
#  for a in 000001 01 02 03 04 05 06 07 08 09 10; do
#  for a in 10 11 12 13 14 15 16 17 18 19 20; do
#  for a in 90 89 88 87 86 85 84 83 82 81 80; do
    wa=0.$a
    w=$a
    ngram -order 3 -unk -map-unk "$oov_symbol" -lm data/local/lm/giga.arpa.gz -mix-lm data/local/lm/srilm.arpa \
          -lambda $wa -write-lm data/local/lm/lm.3gram.interpolated.${w}.gz
    echo -n "interpolation weight $wa, $wb "
    ngram -order 3 -unk -map-unk "$oov_symbol" -lm data/local/lm/lm.3gram.interpolated.${w}.gz \
      -ppl data/local/lm/dev | paste -s
  done
  exit
fi

if [ $stage -le 4 ]; then
  dir=data/lang_test_giga/
  w=04
  zcat data/local/lm/lm.3gram.interpolated.${w}.gz > data/local/lm/merged.arpa

  arpafile=data/local/lm/merged.arpa
  mkdir -p $dir
  cp data/lang/* $dir/ -r

  arpa2fst --disambig-symbol=#0 --read-symbol-table=$dir/words.txt $arpafile $dir/G.fst

  echo  "Checking how stochastic G is (the first of these numbers should be small):"
  fstisstochastic $dir/G.fst

  echo "First few lines of lexicon FST:"
  fstprint --isymbols=$dir/phones.txt --osymbols=$dir/words.txt $dir/L.fst  | head

  echo Performing further checks

# Checking that G.fst is determinizable.
  fstdeterminize $dir/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
  fstdeterminize $dir/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
  fsttablecompose $dir/L_disambig.fst $dir/G.fst | \
     fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
  fsttablecompose $dir/L_disambig.fst $dir/G.fst | \
     fstisstochastic || echo "[log:] LG is not stochastic"

fi
