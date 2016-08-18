
# Model Details

This model was created by using the following command line:

```
./bin/hyper-cli.py --train data/yago3_mte10_5k/yago3_mte10-train.tsv.gz --epochs 100 --optimizer adagrad --lr 0.1 --batches 10 --model ComplEx --similarity dot --margin 1 --entity-embedding-size 5 --save model_farm/yago3_mte10_complex_5_dot/yago3
```
        