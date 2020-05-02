#! /bin/bash

# $1 ../data/l1000/max10/
# $2 num of files, 5

idx=0
while [ ${idx} -lt $2 ]; do
  echo $1${idx}"_neg.txt"
  python preprocess.py --train $1${idx}"_neg.txt" --split 10 --jobs 16
  mkdir $1${idx}"_neg"
  mv $1"/"tensors* $1${idx}"_neg"
  idx=$(( ${idx}+1 ))
done
