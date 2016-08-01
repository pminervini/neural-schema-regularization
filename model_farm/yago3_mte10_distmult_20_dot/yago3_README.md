
        # Model Details

        This model was created by using the following command line:

        ```
        ./bin/hyper-cli.py --train data/yago3_mte10_5k/yago3_mte10-train.tsv.gz --epochs 500 --optimizer adagrad --lr 0.1 --batches 10 --model DistMult --similarity dot --margin 1 --entity-embedding-size 20 --save model_farm/yago3_mte10_distmult_20_dot/yago3
        ```
        