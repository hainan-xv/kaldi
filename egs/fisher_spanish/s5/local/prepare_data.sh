#!/bin/bash
stage=3
speechpath=$1
transpath=$2

tmpdir=data/local/texttmp
dir=data/train
mkdir -p $tmpdir
mkdir -p $dir

if [ $stage -le 1 ]; then # text
cat $transpath/fisher_spa_tr/data/transcripts/* | awk -F '\t' 'NF==13' | gzip > $tmpdir/all_text.gz

  zcat $tmpdir/all_text.gz | tail -n +2 | sed "s=_fsp.sph==g" | awk -F '\t' '{side="A";if($2==1){side="B"}printf("%s-%s_%06d_%06d\n",$1,side,int($3*100),int($4*100))}' > $tmpdir/utt-ids
  zcat $tmpdir/all_text.gz | tail -n +2 | awk -F '\t' '{print $8}' > $tmpdir/rawtext.txt

  cat $tmpdir/rawtext.txt | \
      tr "A-Z" "a-z" | \
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
      sed "s=NOISE=<noise>=g" | \
      sed "s=LAUGH=<laugh>=g" \
      > $tmpdir/cleantext.txt
    
  paste $tmpdir/utt-ids $tmpdir/cleantext.txt > $dir/text
fi # generate data/train/text

if [ $stage -le 2 ]; then # wav.scp
  find $speechpath/ -name "*.sph" > $tmpdir/sph.flist
  cat $tmpdir/sph.flist | awk -F '/' '{print $NF}' | sed "s=_fsp.sph==g" > $tmpdir/rec-ids
  paste $tmpdir/rec-ids $tmpdir/sph.flist | \
    awk -v kaldi=$KALDI_ROOT '{printf("%s-A %s/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 %s |\n",$1,kaldi,$2); printf("%s-B %s/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 2 %s |\n",$1,kaldi,$2)}' > $dir/wav.scp
fi  

if [ $stage -le 3 ]; then # segment and reco2file_and_channel and utt2spk
  cat $tmpdir/utt-ids | awk -F '_' '{print $0, $1"_"$2"_"$3, $4 / 100, $5 / 100}' > $dir/segments
  cat $tmpdir/rec-ids | awk -F '-' '{print $0"-A", $1" A"; print $0"-B", $1" B"}' > $dir/reco2file_and_channel
  cat $tmpdir/utt-ids | awk -F '_' '{print $0, $1"_"$2"_"$3}' > $dir/utt2spk
fi
