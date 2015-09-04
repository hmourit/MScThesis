#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q R4hi.q,R8hi.q,R32hi.q,R128hi.q
#$ -N RFE
#$ -M hmourit+mumak@gmail.com
#$ -m be
#$ -o $HOME/bucket/logs/$JOB_ID.txt
##$ -e $HOME/bucket/logs/$JOB_ID.txt

python $HOME/MScThesis/rfe.py --data $1 --target $2 --clf $3 --n-folds $4 -v --n-iter 20