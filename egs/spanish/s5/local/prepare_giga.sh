#!/bin/bash


datadir=/export/corpora5/LDC/LDC2011T12

dir=data/local/gigatmp/

mkdir -p $dir

(
for i in $datadir/data/*/*.gz; do
  zcat $i | perl -p -e "s=\n= =g" | sed "s=<P>=\n=g" | sed "s=</P>.*==g" | grep -v "^<" 
done
) | head -n 1000000 | gzip > $dir/all.gz # TODO

local/make_data_giga.sh $dir/all.gz
