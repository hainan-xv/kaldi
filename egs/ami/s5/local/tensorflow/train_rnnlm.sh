#!/bin/bash

#set -v
set -e

train_text=data/ihm/train/text
nwords=9997

. path.sh
. cmd.sh

. utils/parse_options.sh

if [ $# != 1 ]; then
   echo "Usage: $0 [options] <dest-dir>"
   echo "For options, see top of script file"
   exit 1;
fi

dir=$1
srcdir=data/local/dict

mkdir -p $dir

cat $srcdir/lexicon.txt | awk '{print $1}' | grep -v -w '!SIL' > $dir/wordlist.all

# Get training data with OOV words (w.r.t. our current vocab) replaced with <unk>.
cat $train_text | awk -v w=$dir/wordlist.all \
  'BEGIN{while((getline<w)>0) v[$1]=1;}
  {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}'|sed 's/ $//g' \
  | perl -e ' use List::Util qw(shuffle); @A=<>; print join("", shuffle(@A)); ' \
  | gzip -c > $dir/all.gz

echo "Splitting data into train and validation sets."
heldout_sent=10000
gunzip -c $dir/all.gz | head -n $heldout_sent > $dir/valid.in # validation data
gunzip -c $dir/all.gz | tail -n +$heldout_sent > $dir/train.in # training data


cat $dir/train.in $dir/wordlist.all | grep -v '</s>' | grep -v '<s>' | \
  awk '{ for(x=1;x<=NF;x++) count[$x]++; } END{for(w in count){print count[w], w;}}' | \
  sort -nr > $dir/unigram.counts

total_nwords=`wc -l $dir/unigram.counts | awk '{print $1}'`

head -$nwords $dir/unigram.counts | awk '{print $2}' | tee $dir/wordlist.rnn | awk '{print NR-1, $1}' > $dir/wordlist.rnn.id

tail -n +$nwords $dir/unigram.counts > $dir/unk_class.counts

for type in train valid; do
  mv $dir/$type.in $dir/$type
done

# Now randomize the order of the training data.
cat $dir/train | awk -v rand_seed=$rand_seed 'BEGIN{srand(rand_seed);} {printf("%f\t%s\n", rand(), $0);}' | \
 sort | cut -f 2 > $dir/foo
mv $dir/foo $dir/train

# OK we'll train the RNNLM on this data.

touch $dir/unk.probs  # dummy file, not used for cued-rnnlm

echo "data preparation finished"

