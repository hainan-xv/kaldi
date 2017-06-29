#!/bin/bash
stage=1
speechpath=$1
transpath=$2

tmpdir=data/local/texttmp
mkdir -p $tmpdir

find $transpath/ -name "*.tdf" | awk -F "/" '{print $NF}' | sed "s=.tdf==g" > $tmpdir/filelist.txt

cat $tmpdir/filelist.txt | shuf --random-source=$tmpdir/filelist.txt > $tmpdir/filelist.shuf

head -n 700 $tmpdir/filelist.shuf > $tmpdir/filelist.train
tail -n +701 $tmpdir/filelist.shuf > $tmpdir/filelist.test
local/make_data.sh $speechpath $transpath $tmpdir/filelist.train data/train data/local/traintmp
local/make_data.sh $speechpath $transpath $tmpdir/filelist.test data/test data/local/testtmp
