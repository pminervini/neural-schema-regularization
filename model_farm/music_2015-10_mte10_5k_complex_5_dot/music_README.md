
# Model Details

This model was created by using the following command line:

```
./bin/hyper-cli.py --train data/music_2015-10_mte10_5k/music_2015-10_mte10-train.tsv.gz --epochs 100 --optimizer adagrad --lr 0.1 --batches 10 --model ComplEx --similarity dot --margin 1 --entity-embedding-size 5 --save model_farm/music_2015-10_mte10_5k_complex_5_dot/music
```
        