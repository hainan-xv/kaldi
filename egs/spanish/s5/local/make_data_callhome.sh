#!/bin/bash
stage=1
speechpath=$1
transpath=$2
filelist=$3
dir=$4
tmpdir=$5

export LC_ALL=C

mkdir -p $tmpdir
mkdir -p $dir

cat $filelist > $tmpdir/tdf.flist
transfilelist=`cat $tmpdir/tdf.flist`

if [ $stage -le 1 ]; then # text
  grep . $transfilelist | gzip > $tmpdir/raw.gz
  grep . $transfilelist | iconv -f iso-8859-1 -t utf-8 | sed "s=^[^ ]*\/sp=sp=g" | sed "s=.txt:= =g" | sed "s=:==g" | gzip > $tmpdir/all_text.gz

  zcat $tmpdir/all_text.gz | awk '{side=substr($4,0,1);printf("%s-%s_%06d_%06d\n",$1,side,int($2*100),int($3*100))}' > $tmpdir/utt-ids

#  cat $tmpdir/utt-ids | sort | uniq -c | sort -k1n | awk '$1!=1{print $2}' > $tmpdir/bad-utt-ids
#  cat $tmpdir/utt-ids | awk -F '_' '$(NF-1)==$NF{print $0}' >> $tmpdir/bad-utt-ids

  zcat $tmpdir/all_text.gz | cut -d ' ' -f5- > $tmpdir/rawtext.txt

  cat $tmpdir/rawtext.txt | \
      tr "A-Z" "a-z" | \
      sed "s={=<=g" | \
      sed "s=}=>=g" | \
      sed "s=\[=<=g" | \
      sed "s=\]=>=g" | \
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
      sed "s=<scream>=LAUGH=g" | \
      sed "s=<sigh>=LAUGH=g" | \
      sed "s=<sneeze>=LAUGH=g" | \
      sed "s=<sniff>=LAUGH=g" | \
      sed "s=<breath>=NOISE=g" | \
      sed "s=<breath=NOISE=g" | \
      sed "s=<kiss=NOISE=g" | \
      sed "s=<whisitle>=NOISE=g" | \
      sed "s=<cough>=NOISE=g" | \
      sed "s=<whisitling>=NOISE=g" | \
      sed "s=</breath>=NOISE=g" | \
      sed "s=<breath/>=NOISE=g" | \
      sed "s=<sneeze/>=NOISE=g" | \
      sed "s=<laugh>=LAUGH=g" | \
      sed "s=<laughter>=LAUGH=g" | \
      sed "s=</laugh>=LAUGH=g" | \
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
      sed "s=LAUGH= <laugh> =g" \
      > $tmpdir/cleantext.txt
#  paste $tmpdir/utt-ids $tmpdir/cleantext.txt | awk 'NF==1' | awk '{print $1}' >> $tmpdir/bad-utt-ids

  rm $tmpdir/bad-utt-ids
  touch $tmpdir/bad-utt-ids

  paste $tmpdir/utt-ids $tmpdir/cleantext.txt | sort | grep -v -f $tmpdir/bad-utt-ids > $dir/text
  paste $tmpdir/utt-ids $tmpdir/cleantext.txt | sort | awk 'BEGIN{last=""}{if($1!=last) {print$0; last=$1};}' | grep -v -f $tmpdir/bad-utt-ids > $dir/text
#  paste $tmpdir/utt-ids $tmpdir/cleantext.txt | awk 'NF>1' | sort | awk 'BEGIN{last=""}{if($1!=last) {print$0; last=$1};}' | grep -v -f $tmpdir/bad-utt-ids > $dir/text

  cat $dir/text | awk '{print $1}' | awk -F '_' '{print$1"_"$2}' | sort -u > $tmpdir/good.ids

fi # generate data/train/text

if [ $stage -le 2 ]; then # wav.scp
  find $speechpath/ -name "*.SPH" > $tmpdir/sph.flist
#  find $speechpath/ -name "*.sph" > $tmpdir/sph.flist
  cat $tmpdir/sph.flist | awk -F '/' '{print $NF}' | sed "s=.SPH==g" | tr A-Z a-z > $tmpdir/rec-ids
  paste $tmpdir/rec-ids $tmpdir/sph.flist | \
    awk -v kaldi=$KALDI_ROOT '{printf("%s-A %s/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 %s |\n",$1,kaldi,$2); printf("%s-B %s/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 2 %s |\n",$1,kaldi,$2)}' | \
    sort | awk 'BEGIN{last=""}{if($1!=last) {print$0; last=$1};}' | grep -v -f $tmpdir/bad-utt-ids | grep -f $tmpdir/good.ids > $dir/wav.scp
fi  

if [ $stage -le 3 ]; then # segment and reco2file_and_channel and utt2spk
  cat $tmpdir/utt-ids | awk -F '_' '{print $0, $1"_"$2, $3 / 100, $4 / 100}' | sort | awk 'BEGIN{last=""}{if($1!=last) {print$0; last=$1};}' | grep -v -f $tmpdir/bad-utt-ids > $dir/segments
  cat $tmpdir/rec-ids | awk -F '-' '{print $0"-A", $1" A"; print $0"-B", $1" B"}' | sort | grep -v -f $tmpdir/bad-utt-ids | awk 'BEGIN{last=""}{if($1!=last) {print$0; last=$1};}' | grep -f $tmpdir/good.ids > $dir/reco2file_and_channel
  cat $tmpdir/utt-ids | awk -F '_' '{print $0, $1"_"$2}' | sort -u | grep -v -f $tmpdir/bad-utt-ids > $dir/utt2spk
fi

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

#utils/fix_data_dir.sh $dir/
