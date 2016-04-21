# YAGO 2

Data downloaded on 01/04/2016 by using the following command:

```bash
$ wget -c http://resources.mpi-inf.mpg.de/yago-naga/amie/data/yago2/yago2core_facts.clean.notypes.tsv.7z
--2016-04-02 01:14:27--  http://resources.mpi-inf.mpg.de/yago-naga/amie/data/yago2/yago2core_facts.clean.notypes.tsv.7z
Resolving resources.mpi-inf.mpg.de (resources.mpi-inf.mpg.de)... 139.19.86.89
Connecting to resources.mpi-inf.mpg.de (resources.mpi-inf.mpg.de)|139.19.86.89|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 8165115 (7.8M) [text/tab-separated-values]
Saving to: ‘yago2core_facts.clean.notypes.tsv.7z’

yago2core_facts.clean.noty 100%[=====================================>]   7.79M  1.02MB/s    in 6.9s    

2016-04-02 01:14:34 (1.13 MB/s) - ‘yago2core_facts.clean.notypes.tsv.7z’ saved [8165115/8165115]

$ md5sum yago2core_facts.clean.notypes.tsv.7z
9b4de2c2d70dac8f4f03e962e811f61b  yago2core_facts.clean.notypes.tsv.7z
```

Training, validation and test sets were created as follows:

```bash
$ mkdir tmp
$ cd tmp/
$ 7z e ../yago2core_facts.clean.notypes.tsv.7z

7-Zip [64] 15.09 beta : Copyright (c) 1999-2015 Igor Pavlov : 2015-10-16
p7zip Version 15.09 beta (locale=en_US.utf8,Utf16=on,HugeFiles=on,64 bits,4 CPUs Intel(R) Core(TM) i7-3537U CPU @ 2.00GHz (306A9),ASM,AES-NI)

Scanning the drive for archives:
1 file, 8165115 bytes (7974 KiB)

Extracting archive: ../yago2core_facts.clean.notypes.tsv.7z
--
Path = ../yago2core_facts.clean.notypes.tsv.7z
Type = 7z
Physical Size = 8165115
Headers Size = 168
Method = LZMA:24
Solid = -
Blocks = 1

Everything is Ok                        

Size:       46458484
Compressed: 8165115
$ ls
yago2core_facts.clean.notypes.tsv
$ wc -l *.tsv
948358 yago2core_facts.clean.notypes.tsv
$ ../tools/splitter.py --kb yago2core_facts.clean.notypes.tsv --train yago2-train.txt --validation yago2-valid.txt --validation-size 1000 --test yago2-test.txt --test-size 1000 --seed 0
DEBUG:root:Importing the Knowledge Graph ..
DEBUG:root:Number of triples in the Knowledge Graph: 948358
DEBUG:root:Generating a random permutation of RDF triples ..
DEBUG:root:Building the training, validation and test sets ..
DEBUG:root:Saving ..
$ wc -l *.txt
1000 yago2-test.txt
946358 yago2-train.txt
1000 yago2-valid.txt
948358 total
$ mv *.txt ..
$ cd ..
$ gzip -9 yago2-*.txt
$ rm -rf tmp/
```
