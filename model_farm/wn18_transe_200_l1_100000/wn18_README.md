
# Model Details

This model was created by using the following command line:

```
./bin/hyper-cli.py --train data/wn18/wordnet-mlj12-train.txt --epochs 100 --optimizer adagrad --lr 0.1 --batches 10 --model TransE --similarity l1 --margin 1 --entity-embedding-size 200 --rules ./data/wn18/rules/wn18-rules.json.gz --rules-top 10 --sample-facts 1 --rules-lambda 100000 --save model_farm/wn18_transe_200_l1_100000/wn18
```
        