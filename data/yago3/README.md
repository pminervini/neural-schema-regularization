# YAGO 3

Data downloaded on 28/02/2016 by using the following command:

```bash
$ wget -c http://resources.mpi-inf.mpg.de/yago-naga/yago/download/yago/yagoFacts.tsv.7z
--2016-02-28 02:38:36--  http://resources.mpi-inf.mpg.de/yago-naga/yago/download/yago/yagoFacts.tsv.7z
Resolving resources.mpi-inf.mpg.de (resources.mpi-inf.mpg.de)... 139.19.86.89
Connecting to resources.mpi-inf.mpg.de (resources.mpi-inf.mpg.de)|139.19.86.89|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 103497683 (99M) [text/tab-separated-values]
Saving to: ‘yagoFacts.tsv.7z’

yagoFacts.tsv.7z           100%[=====================================>]  98.70M   885KB/s    in 2m 9s   

2016-02-28 02:40:45 (786 KB/s) - ‘yagoFacts.tsv.7z’ saved [103497683/103497683]

$ md5sum yagoFacts.tsv.7z
5775c0d55750025c0579160e303ca507  yagoFacts.tsv.7z
```

Training, validation and test sets were created as follows:

```bash
$ mkdir tmp
$ cd tmp/
$ 7z e ../yagoFacts.tsv.7z

7-Zip [64] 9.20  Copyright (c) 1999-2010 Igor Pavlov  2010-11-18
p7zip Version 9.20 (locale=en_US.utf8,Utf16=on,HugeFiles=on,4 CPUs)

Processing archive: ../yagoFacts.tsv.7z

Extracting  yagoFacts.tsv

Everything is Ok

Size:       423438221
Compressed: 103497683
$ wc -l yagoFacts.tsv
5628167 yagoFacts.tsv
$ ../tools/splitter.py --kb yagoFacts.tsv --train yago3-train.txt --validation yago3-valid.txt --validation-size 50000 --test yago3-test.txt --test-size 50000 --seed 0
DEBUG:root:Importing the Knowledge Graph ..
DEBUG:root:Number of triples in the Knowledge Graph: 5628166
DEBUG:root:Generating a random permutation of RDF triples ..
DEBUG:root:Building the training, validation and test sets ..
DEBUG:root:Saving ..
pasquale@koeln:~/insight/workspace/hyper/data/yago3/tmp$ wc -l *.txt
    50000 yago3-test.txt
  5528166 yago3-train.txt
    50000 yago3-valid.txt
  5628166 total
$ mv *.txt ..
$ cd ..
$ gzip -9 yago3-*.txt
$ rm -rf tmp/
```
