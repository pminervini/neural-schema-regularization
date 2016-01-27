cat ../original/*.txt | perl -MList::Util=shuffle -e 'srand(1); print shuffle(<STDIN>);' > all-shuffled.txt
