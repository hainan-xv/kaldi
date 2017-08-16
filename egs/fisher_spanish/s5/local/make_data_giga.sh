#!/bin/bash

[ -f ./path.sh ] && . ./path.sh

textgz=$1
tmpdir=data/local/gigatmp
stage=1

. parse_options.sh || exit 1;

mkdir -p $tmpdir

if [ $stage -le 1 ]; then
      zcat $textgz | \
      tr "A-Z" "a-z" | \
      sed "s/´/'/g" | \
      sed "s/¡/i/g" | \
      sed "s/è/é/g" | \
      sed "s/ì/í/g" | \
      sed "s/·//g" | \
      sed "s/ç//g" | \
      sed "s/¨//g" | \
      sed "s/Ñ/ñ/g" | \
      sed "s/à/á/g" | \
      sed "s/Á/á/g" | \
      sed "s/É/é/g" | \
      sed "s/Í/í/g" | \
      sed "s/Ó/ó/g" | \
      sed "s/Ú/ú/g" | \
      sed "s=<laugh> *</laugh>=LAUGH=g" | \
      sed "s=< fore=<fore=g" |\
      sed "s=<foreign lan=<foreignlan=g" | \
      sed "s=<foreigh lan=<foreignlan=g" | \
      sed "s=<fore[^ ]*>==g" | \
      sed "s=</fore[^ ]*>==g" | \
      sed "s=<lname>==g" | \
      sed "s=<lname/>==g" | \
      sed "s=</lname>==g" | \
      sed "s=<background>=NOISE=g" | \
      sed "s=</background>=NOISE=g" | \
      sed "s=<breath>=NOISE=g" | \
      sed "s=</breath>=NOISE=g" | \
      sed "s=<breath/>=NOISE=g" | \
      sed "s=<sneeze/>=NOISE=g" | \
      sed "s=<laugh>=NOISE=g" | \
      sed "s=</laugh>=NOISE=g" | \
      sed "s=<cough>=NOISE=g" | \
      sed "s=<cough/>=NOISE=g" | \
      sed "s=<lipsmack/>=NOISE=g" | \
#    sed "s=english\"=english\" =g" |\
      sed "s= [^ ]*/[^ ]* = =g" |\
      sed "s= [^ ]*<[^ ]* = =g" |\
      sed "s= [^ ]*>[^ ]* = =g" |\
      sed "s= [^ ]*>[^ ]*$= =g" |\
      sed "s=¿==g" |\
      sed "s=[[:punct:]]==g" |\
      awk '{for(i=1;i<=NF;i++) printf("%s ",$i); print""}' |\
      sed "s=LAUGH LAUGH=LAUGH=g" | \
      sed "s=NOISE NOISE=NOISE=g" | \
      sed "s=NOISE= <noise> =g" | \
      sed "s=LAUGH= <laugh> =g" | \
      grep . \
      | gzip > $tmpdir/cleantext.txt.gz
fi

if [ $stage -le 2 ]; then
#zcat $tmpdir/cleantext.txt.gz | 
  zcat $tmpdir/cleantext.txt.gz | head -n 10000000 > $tmpdir/cleantext.head.txt
  cat $tmpdir/cleantext.head.txt | awk '{print "id: ",$0}' > $tmpdir/text
fi
