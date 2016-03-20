#!/bin/bash

reverse=false
replace=

. utils/parse_options.sh || exit 1;

file=$1
a=$2
b=$3

#set -x
#cat $file | sed "s= $==g" | awk -v a="$a" -v b="$b" 'BEGIN{p=0}{if($0==a) p=1; if (p==1) print; if($0==b) p=0;}'

if [ $reverse == "false" ]; then
  cat $file | awk -v a="$a" -v b="$b" 'BEGIN{p=0}{for(i=1;i<=NF;i++){if($i==a) p=1; if (p==1) printf("%s ",$i); if($i==b) exit 0;}print""}' | grep .
elif [ "$replace" = "" ]; then
  cat $file | awk -v a="$a" -v b="$b" 'BEGIN{p=0}{for(i=1;i<=NF;i++){if($i==a) p=1; if (p==0) printf("%s ",$i); if($i==b) p=0;}print""}' | grep .
else
  cat $file | awk -v a="$a" -v b="$b" -v replace="$(cat $replace)" 'BEGIN{p=0}{for(i=1;i<=NF;i++){if($i==a) {print replace; p=1}; if (p==0) printf("%s ",$i); if($i==b) p=0;}print""}' | grep .
fi

  
