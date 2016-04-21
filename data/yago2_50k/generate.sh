#!/usr/bin/env bash

rm yago2-*.txt yago2-*.txt.gz

rm -rf tmp
mkdir tmp
cd tmp

7z e ../yago2core_facts.clean.notypes.tsv.7z
wc -l *.tsv
../tools/splitter.py --kb yago2core_facts.clean.notypes.tsv --train yago2-train.txt --validation yago2-valid.txt --validation-size 50000 --test yago2-test.txt --test-size 50000 --seed 0
wc -l *.txt

mv *.txt ..
cd ..
gzip -9 yago2-*.txt

md5sum yago2-*.txt.gz
rm -rf tmp/
