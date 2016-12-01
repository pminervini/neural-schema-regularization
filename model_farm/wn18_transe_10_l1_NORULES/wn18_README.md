
# Model Details

This model was created by using the following command line:

```
./bin/hyper-cli.py --train data/wn18/wordnet-mlj12-train.txt --epochs 1000 --optimizer adagrad --lr 0.1 --batches 10 --model TransE --similarity l1 --margin 1 --entity-embedding-size 10 --save model_farm/wn18_transe_10_l1_NORULES/wn18
```
        