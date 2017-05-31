#!/bin/bash
# Copyright 2014  Gaurav Kumar.   Apache 2.0
#

if [ -f path.sh ]; then . path.sh; fi

arpa_lm=data/local/lm/3gram-mincount/lm_unpruned.gz
[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

mkdir -p data/lang_nosp_test
cp -r data/lang_nosp/* data/lang_nosp_test

gunzip -c "$arpa_lm" | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=data/lang_nosp_test/words.txt - data/lang_nosp_test/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic data/lang_nosp_test/G.fst

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=data/lang_nosp/phones.txt --osymbols=data/lang_nosp/words.txt data/lang_nosp/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize data/lang_nosp_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize data/lang_nosp_test/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose data/lang_nosp_test/L_disambig.fst data/lang_nosp_test/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/lang_nosp/L_disambig.fst data/lang_nosp_test/G.fst | \
   fstisstochastic || echo "[log:] LG is not stochastic"


echo "$0 succeeded"
