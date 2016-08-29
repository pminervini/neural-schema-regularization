
# Model Details

This model was created by using the following command line:

```
./bin/hyper-cli.py --train data/music_2015-10_mte10_5k/music_2015-10_mte10-train.tsv.gz --epochs 100 --optimizer adagrad --lr 0.1 --batches 10 --model TransE --similarity l1 --margin 1 --entity-embedding-size 10 --save model_farm/music_2015-10_mte10_5k_transe_10_l1/music
```
        