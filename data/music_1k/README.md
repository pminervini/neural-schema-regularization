# DBpedia 2015-04 (Music)

Data downloaded on 28/02/2016 from http://dbpedia.org/Downloads2015-04 by using the following command:

```bash
$ wget -c --quiet http://downloads.dbpedia.org/2015-04/core-i18n/en/mappingbased-properties_en.nt.bz2
$ wget -c --quiet http://downloads.dbpedia.org/2015-04/core-i18n/en/mappingbased-properties-errors-unredirected_en.nt.bz2
$ wget -c --quiet http://downloads.dbpedia.org/2015-04/core-i18n/en/specific-mappingbased-properties_en.nt.bz2
$ du -hs *.bz2
266M	mappingbased-properties_en.nt.bz2
416K	mappingbased-properties-errors-unredirected_en.nt.bz2
6.1M	specific-mappingbased-properties_en.nt.bz2
$ md5sum *.bz2
8407c84d262b573418326bdd8f591b95  mappingbased-properties_en.nt.bz2
cbf72636d8f03a9c027eb8a70a02f50c  mappingbased-properties-errors-unredirected_en.nt.bz2
1904ad5bc4579fd7efe7f40673c32f79  specific-mappingbased-properties_en.nt.bz2
$ bzcat mappingbased-properties_en.nt.bz2 | grep '<http://dbpedia.org/ontology/genre>\|<http://dbpedia.org/ontology/recordLabel>\|<http://dbpedia.org/ontology/associatedMusicalArtist>\|<http://dbpedia.org/ontology/associatedBand>\|<http://dbpedia.org/ontology/musicalArtist>\|<http://dbpedia.org/ontology/musicalBand>\|<http://dbpedia.org/ontology/album>' > music.nt
$ du -hs *.nt
134M	music.nt
$ md5sum music.nt 
f0241e99bde3c880e33180dbfbbab443  music.nt
```

Training, validation and test sets were created as follows:

```bash
$ ./tools/splitter.py --kb music.nt --train music-train.txt --validation music-valid.txt --validation-size 1000 --test music-test.txt --test-size 1000 --seed 0
DEBUG:root:Importing the Knowledge Graph ..
DEBUG:root:Number of triples in the Knowledge Graph: 1025587
DEBUG:root:Generating a random permutation of RDF triples ..
DEBUG:root:Building the training, validation and test sets ..
DEBUG:root:Saving ..
$ md5sum *.txt
79541f6f943b26ab96e850001446373d  music-test.txt
888e635c5a66e6633efa1454dfb0e591  music-train.txt
61f3a4577a7ecd18ec46545b843a4525  music-valid.txt
$ gzip -9 *.txt music.nt
$ zcat music.nt.gz | cut -f 2 -d ' ' | sort | uniq -c
  37989 <http://dbpedia.org/ontology/album>
 116444 <http://dbpedia.org/ontology/associatedBand>
 116428 <http://dbpedia.org/ontology/associatedMusicalArtist>
 447588 <http://dbpedia.org/ontology/genre>
  43196 <http://dbpedia.org/ontology/musicalArtist>
  43196 <http://dbpedia.org/ontology/musicalBand>
 220746 <http://dbpedia.org/ontology/recordLabel>
$
```