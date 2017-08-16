#!/bin/bash
. cmd.sh
. path.sh

text=data/train_fisher/text
text2=data/train_callhome/text
tmpdir=data/local/lm
devsize=10000

arpafile=$tmpdir/srilm.arpa

mkdir -p $tmpdir

cat $text | awk '{for(i=2;i<=NF;i++) printf("%s ",$i); print""}' | shuf --random-source=$text > $tmpdir/clean.txt

head -n $devsize $tmpdir/clean.txt > $tmpdir/dev
tail -n +$[$devsize+1] $tmpdir/clean.txt > $tmpdir/train

cat data/lang_nosp/words.txt | awk '{print $1}' > $tmpdir/wlist

ngram-count -text $tmpdir/train -order 3 -limit-vocab $tmpdir/wlist -kndiscount -lm $arpafile
(
ngram -lm $arpafile -ppl $tmpdir/train
ngram -lm $arpafile -ppl $tmpdir/dev
)

dir=data/lang_nosp_test/

mkdir -p $dir
cp data/lang_nosp/* $dir/ -r

arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang_nosp_test/words.txt $arpafile $dir/G.fst

echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic data/lang_nosp_test/G.fst

echo "First few lines of lexicon FST:"
fstprint --isymbols=data/lang_nosp/phones.txt --osymbols=data/lang_nosp/words.txt data/lang_nosp/L.fst  | head

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
