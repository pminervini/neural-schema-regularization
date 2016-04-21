#!/usr/bin/env bash

rm yago3-*.txt yago3-*.txt.gz

rm -rf tmp
mkdir tmp
cd tmp

7z e ../yagoFacts.tsv.7z
wc -l *.tsv
../tools/splitter.py --kb yagoFacts.tsv --train yago3-train.txt --validation yago3-valid.txt --validation-size 5000 --test yago3-test.txt --test-size 5000 --seed 0
wc -l *.txt

mv *.txt ..
cd ..
gzip -9 yago3-*.txt

md5sum yago3-*.txt.gz
rm -rf tmp/
