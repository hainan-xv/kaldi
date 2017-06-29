#!/bin/bash

export LC_ALL=C


stage=1

lexicon=$1

dir=data/local/dict
tmpdir=data/local/dict/tmp
mkdir -p $tmpdir

if [ $stage -le 1 ]; then
  cat data/train/text | cut -f2- | sed "s= =\n=g" | sort -u | grep -v "<" | grep . > $tmpdir/wordlist.txt

  cat $lexicon/callhome_spanish_lexicon_970908/preferences | iconv -f iso-8859-1 -t utf-8 > local/preferences
  cat $lexicon/callhome_spanish_lexicon_970908/basic_rules | iconv -f iso-8859-1 -t utf-8 > local/basic_rules

  cat $tmpdir/wordlist.txt | local/spron-utf8.pl local/preferences local/basic_rules \
    | cut -f1 | sed -r 's:#\S+\s\S+\s\S+\s\S+\s(\S+):\1:g' | sed "s=ñ=N=g" \
    | awk -F '[/][/]' '{print $1}' | sed 's/./& /g'  > $tmpdir/lexicon.raw
fi

if [ $stage -le 2 ]; then
  paste $tmpdir/wordlist.txt $tmpdir/lexicon.raw > $dir/lexicon.txt
  echo "<unk> SIL" >> $dir/lexicon.txt

  cat $tmpdir/lexicon.raw | sed "s= =\n=g" | sort -u | grep . > $dir/nonsilence_phones.txt
  
  echo "<laugh> LAUGH" >> $dir/lexicon.txt
  echo "<noise> NOISE" >> $dir/lexicon.txt

  touch $dir/extra_questions.txt

  echo "LAUGH" > $dir/silence_phones.txt
  echo "NOISE" >> $dir/silence_phones.txt
  echo "SIL" >> $dir/silence_phones.txt

  echo "SIL" > $dir/optional_silence.txt
fi
