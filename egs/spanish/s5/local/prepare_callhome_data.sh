#!/bin/bash
stage=1
speechpath=$1
transpath=$2

tmpdir=data/local/texttmp_callhome
mkdir -p $tmpdir

find $transpath/ -name "*.txt"  > $tmpdir/filelist.txt

cat $tmpdir/filelist.txt | shuf --random-source=$tmpdir/filelist.txt > $tmpdir/filelist.shuf

head -n 8 $tmpdir/filelist.shuf > $tmpdir/filelist.test
tail -n +9 $tmpdir/filelist.shuf > $tmpdir/filelist.train

local/make_data_callhome.sh $speechpath $transpath $tmpdir/filelist.train data/train_callhome data/local/traintmp.callhome
local/make_data_callhome.sh $speechpath $transpath $tmpdir/filelist.test data/test_callhome data/local/testtmp.callhome

