
# Model Details

This model was created by using the following command line:

```
./bin/hyper-cli.py --train data/wn18/wordnet-mlj12-train.txt --epochs 100 --optimizer adagrad --lr 0.1 --batches 10 --model DistMult --similarity dot --margin 1 --entity-embedding-size 10 --save model_farm/wn18_distmult_10_dot/wn18
```
        