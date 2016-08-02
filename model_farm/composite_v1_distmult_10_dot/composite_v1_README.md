
# Model Details

This model was created by using the following command line:

```
./bin/hyper-cli.py --train data/composite_v1/composite_v1.tsv.gz --epochs 500 --optimizer adagrad --lr 0.1 --batches 10 --model DistMult --similarity dot --margin 1 --entity-embedding-size 10 --save model_farm/composite_v1_distmult_10_dot/composite_v1
```
        