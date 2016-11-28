
# Model Details

This model was created by using the following command line:

```
./bin/hyper-cli.py --train data/wn18/wordnet-mlj12-train.txt --epochs 100 --optimizer adagrad --lr 0.1 --batches 10 --model ComplEx --similarity dot --margin 1 --entity-embedding-size 5 --save model_farm/wn18_complex_5_dot_RULES/wn18 --rules ./data/wn18/rules/wn18-rules.json.gz --rules-top 10 --sample-facts 1 --rules-lambda 10000
```
        