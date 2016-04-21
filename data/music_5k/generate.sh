#!/usr/bin/env bash

rm music-*.txt music-*.txt.gz music.nt

zcat music.nt.gz > music.nt
wc -l *.nt
./tools/splitter.py --kb music.nt --train music-train.txt --validation music-valid.txt --validation-size 5000 --test music-test.txt --test-size 5000 --seed 0

wc -l *.txt
md5sum *.txt

gzip -9 music-*.txt

md5sum music-*.txt.gz
rm music.nt
