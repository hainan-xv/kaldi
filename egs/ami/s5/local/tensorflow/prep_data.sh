#!/bin/bash

#set -v
set -e

train_text=data/ihm/train/text
nwords=9999

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

cat $srcdir/lexicon.txt | awk '{print $1}' | sort -u | grep -v -w '!SIL' > $dir/wordlist.all

# Get training data with OOV words (w.r.t. our current vocab) replaced with <unk>.
cat $train_text | awk -v w=$dir/wordlist.all \
  'BEGIN{while((getline<w)>0) v[$1]=1;}
  {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}' | sed 's=$= </s>=g' \
  | gzip -c > $dir/all.gz

echo "Splitting data into train and validation sets."
heldout_sent=10000
gunzip -c $dir/all.gz | head -n $heldout_sent > $dir/valid.in # validation data
gunzip -c $dir/all.gz | tail -n +$heldout_sent > $dir/train.in # training data


cat $dir/train.in $dir/wordlist.all | \
  awk '{ for(x=1;x<=NF;x++) count[$x]++; } END{for(w in count){print count[w], w;}}' | \
  sort -nr > $dir/unigram.counts

total_nwords=`wc -l $dir/unigram.counts | awk '{print $1}'`

head -$nwords $dir/unigram.counts | awk '{print $2}' | tee $dir/wordlist.rnn | awk '{print NR-1, $1}' > $dir/wordlist.rnn.id

tail -n +$nwords $dir/unigram.counts > $dir/unk_class.counts

for type in train valid; do
  cat $dir/$type.in | awk -v w=$dir/wordlist.rnn 'BEGIN{while((getline<w)>0)d[$1]=1}{for(i=1;i<=NF;i++){if(d[$i]==1){s=$i}else{s="<oos>"} printf("%s ",s)} print""}' > $dir/$type
done

# OK we'll train the RNNLM on this data.

cat $dir/unk_class.counts | awk '{print $2, $1}' > $dir/unk.probs  # dummy file, not used for cued-rnnlm

cp $dir/wordlist.rnn $dir/wordlist.rnn.final

has_oos=`grep "<oos>" $dir/wordlist.rnn.final | wc -l | awk '{print $1}'`
if [ $has_oos == "0" ]; then
#  n=`wc -l $dir/wordlist.rnn.final | awk '{print $1}'`
#  echo n is $n
  echo "<oos>" >> $dir/wordlist.rnn.final
fi


echo "data preparation finished"

