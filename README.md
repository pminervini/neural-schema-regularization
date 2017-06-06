# Regularization of Link Prediction Models

This package implements several state-of-the-art latent factor models for link prediction in Knowledge Graphs, and a set of novel regularizers based on background knowledge.

## Installation

This library is heavily based on Keras and Theano.

For installing the latest version of such libraries, run the following command:

```bash
$ sudo pip3 install --upgrade git+git://github.com/fchollet/keras.git git+git://github.com/Theano/Theano.git
```

Then, for installing this library as a user, run the following command:

```bash
$ python3 -uB setup.py install --user
```

## Running

Usage:

```bash
$ ./bin/hyper-cli.py -h
Using Theano backend.
usage: Latent Factor Models for Knowledge Hypergraphs [-h] --train TRAIN [--validation VALIDATION] [--test TEST] [--seed SEED] [--entity-embedding-size ENTITY_EMBEDDING_SIZE]
                                                      [--predicate-embedding-size PREDICATE_EMBEDDING_SIZE] [--dropout-entity-embeddings DROPOUT_ENTITY_EMBEDDINGS]
                                                      [--dropout-predicate-embeddings DROPOUT_PREDICATE_EMBEDDINGS] [--rules RULES] [--rules-top-k RULES_TOP_K] [--rules-threshold RULES_THRESHOLD]
                                                      [--rules-max-length RULES_MAX_LENGTH] [--sample-facts SAMPLE_FACTS] [--rules-lambda RULES_LAMBDA] [--robust] [--robust-alpha ROBUST_ALPHA]
                                                      [--robust-beta ROBUST_BETA] [--sort] [--frequency-embedding-lengths FREQUENCY_EMBEDDING_LENGTHS [FREQUENCY_EMBEDDING_LENGTHS ...]]
                                                      [--frequency-cutoffs FREQUENCY_CUTOFFS [FREQUENCY_CUTOFFS ...]] [--frequency-mask-type {1,2,3}] [--entity-rank ENTITY_RANK]
                                                      [--predicate-rank PREDICATE_RANK] [--model MODEL] [--similarity SIMILARITY] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--batches BATCHES]
                                                      [--margin MARGIN] [--loss LOSS] [--negatives NEGATIVES] [--predicate-l1 PREDICATE_L1] [--predicate-l2 PREDICATE_L2] [--predicate-nonnegative]
                                                      [--hidden-size HIDDEN_SIZE] [--raw] [--filtered] [--visualize] [--optimizer OPTIMIZER] [--lr LR] [--optimizer-momentum OPTIMIZER_MOMENTUM]
                                                      [--optimizer-decay OPTIMIZER_DECAY] [--optimizer-nesterov] [--optimizer-epsilon OPTIMIZER_EPSILON] [--optimizer-rho OPTIMIZER_RHO]
                                                      [--optimizer-beta1 OPTIMIZER_BETA1] [--optimizer-beta2 OPTIMIZER_BETA2] [--fast-eval] [--save SAVE]

optional arguments:
  -h, --help                                                                                   show this help message and exit
  --train TRAIN
  --validation VALIDATION
  --test TEST
  --seed SEED                                                                                  Seed for the PRNG
  --entity-embedding-size ENTITY_EMBEDDING_SIZE                                                Size of entity embeddings
  --predicate-embedding-size PREDICATE_EMBEDDING_SIZE                                          Size of predicate embeddings
  --dropout-entity-embeddings DROPOUT_ENTITY_EMBEDDINGS                                        Dropout after the entity embeddings layer
  --dropout-predicate-embeddings DROPOUT_PREDICATE_EMBEDDINGS                                  Dropout after the predicate embeddings layer
  --rules RULES                                                                                JSON document containing the rules extracted from the KG
  --rules-top-k RULES_TOP_K                                                                    Top-k rules to consider during the training process
  --rules-threshold RULES_THRESHOLD                                                            Only show the rules with a score above the given threshold
  --rules-max-length RULES_MAX_LENGTH                                                          Maximum (body) length for the considered rules
  --sample-facts SAMPLE_FACTS                                                                  Fraction of (randomly sampled) facts to use during training
  --rules-lambda RULES_LAMBDA                                                                  Weight of the Rules-related regularization term
  --robust                                                                                     Robust Ranking
  --robust-alpha ROBUST_ALPHA                                                                  Robust Ranking, Alpha parameter
  --robust-beta ROBUST_BETA                                                                    Robust Ranking, Beta parameter
  --sort                                                                                       Sort entities according to their frequency in the training set
  --frequency-embedding-lengths FREQUENCY_EMBEDDING_LENGTHS [FREQUENCY_EMBEDDING_LENGTHS ...]  Frequency-based embedding lengths
  --frequency-cutoffs FREQUENCY_CUTOFFS [FREQUENCY_CUTOFFS ...]                                Frequency cutoffs
  --frequency-mask-type {1,2,3}                                                                Frequency-based embedding lengths - Mask type
  --entity-rank ENTITY_RANK                                                                    Rank of the entity embeddings matrix
  --predicate-rank PREDICATE_RANK                                                              Rank of the predicate embeddings matrix
  --model MODEL                                                                                Name of the model to use
  --similarity SIMILARITY                                                                      Name of the similarity function to use (if distance-based model)
  --epochs EPOCHS                                                                              Number of training epochs
  --batch-size BATCH_SIZE                                                                      Batch size
  --batches BATCHES                                                                            Number of batches
  --margin MARGIN                                                                              Margin to use in the hinge loss
  --loss LOSS                                                                                  Loss function to be used (e.g. hinge, logistic)
  --negatives NEGATIVES                                                                        Method for generating the negative examples (e.g. corrupt, lcwa, schema, bernoulli)
  --predicate-l1 PREDICATE_L1                                                                  L1 Regularizer on the Predicate Embeddings
  --predicate-l2 PREDICATE_L2                                                                  L2 Regularizer on the Predicate Embeddings
  --predicate-nonnegative                                                                      Enforce a non-negativity constraint on the predicate embeddings
  --hidden-size HIDDEN_SIZE                                                                    Dimension of the hidden layer (used by e.g. the ER-MLP model
  --raw                                                                                        Evaluate the model in the raw setting
  --filtered                                                                                   Evaluate the model in the filtered setting
  --visualize                                                                                  Visualize the embeddings
  --optimizer OPTIMIZER                                                                        Optimization algorithm to use - sgd, adagrad, adadelta, rmsprop, adam, adamax
  --lr LR, --optimizer-lr LR                                                                   Learning rate
  --optimizer-momentum OPTIMIZER_MOMENTUM                                                      Momentum parameter of the SGD optimizer
  --optimizer-decay OPTIMIZER_DECAY                                                            Decay parameter of the SGD optimizer
  --optimizer-nesterov                                                                         Applies Nesterov momentum to the SGD optimizer
  --optimizer-epsilon OPTIMIZER_EPSILON                                                        Epsilon parameter of the adagrad, adadelta, rmsprop, adam and adamax optimizers
  --optimizer-rho OPTIMIZER_RHO                                                                Rho parameter of the adadelta and rmsprop optimizers
  --optimizer-beta1 OPTIMIZER_BETA1                                                            Beta1 parameter for the adam and adamax optimizers
  --optimizer-beta2 OPTIMIZER_BETA2                                                            Beta2 parameter for the adam and adamax optimizers
  --fast-eval                                                                                  Fast Evaluation
  --save SAVE     
```

The following command trains a Translating Embeddings model (in its L1 formulation) on the FB15k dataset: the training process uses AdaGrad [2] with a 0.1 learning rate and a 100 embedding size, and then returns the Mean Rank and Hits@10 metrics on both the validation and the test set.

```bash
$ ./bin/hyper-cli.py --train data/fb15k/freebase_mtr100_mte100-train.txt --valid data/fb15k/freebase_mtr100_mte100-valid.txt --test data/fb15k/freebase_mtr100_mte100-test.txt --epochs 100 --optimizer adagrad --lr 0.1 --batches 10 --model TransE --similarity L1 --entity-embedding-size 100 --predicate-embedding-size 100

Using Theano backend.
INFO:root:Samples: 483142, no. batches: 10 -> batch size: 48315
INFO:root:Epoch no. 1 of 100 (samples: 483142)
INFO:root:Loss: 0.2919 +/- 0.2563
INFO:root:Epoch no. 2 of 100 (samples: 483142)
INFO:root:Loss: 0.0464 +/- 0.0027
...
INFO:root:Epoch no. 99 of 100 (samples: 483142)
INFO:root:Loss: 0.0101 +/- 0.0004
INFO:root:Epoch no. 100 of 100 (samples: 483142)
INFO:root:Loss: 0.0099 +/- 0.0004
INFO:root:### MICRO (validation raw):
INFO:root:      -- left   >> mean: 272.80402, median: 18.0, hits@10: 40.446%
INFO:root:      -- right  >> mean: 170.54526, median: 11.0, hits@10: 48.964%
INFO:root:      -- global >> mean: 221.67464, median: 14.0, hits@10: 44.705%
INFO:root:### MICRO (validation filtered):
INFO:root:      -- left   >> mean: 100.9935, median: 6.0, hits@10: 59.958%
INFO:root:      -- right  >> mean: 69.56062, median: 4.0, hits@10: 66.312%
INFO:root:      -- global >> mean: 85.27706, median: 5.0, hits@10: 63.135%
INFO:root:### MICRO (test raw):
INFO:root:      -- left   >> mean: 270.65557, median: 18.0, hits@10: 40.709%
INFO:root:      -- right  >> mean: 177.68695, median: 12.0, hits@10: 48.096%
INFO:root:      -- global >> mean: 224.17126, median: 14.0, hits@10: 44.402%
INFO:root:### MICRO (test filtered):
INFO:root:      -- left   >> mean: 100.17838, median: 6.0, hits@10: 60.199%
INFO:root:      -- right  >> mean: 71.48272, median: 4.0, hits@10: 65.535%
INFO:root:      -- global >> mean: 85.83055, median: 5.0, hits@10: 62.867%
```

Similarly, we can run an experiment on the WN18 dataset:

```bash
$ ./bin/hyper-cli.py --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt --epochs 1000 --optimizer adagrad --lr 0.1 --batches 10 --model TransE --similarity L1 --entity-embedding-size 20 --predicate-embedding-size 20 --margin 2

Using Theano backend.
INFO:root:Samples: 141442, no. batches: 10 -> batch size: 14145
INFO:root:Epoch no. 1 of 1000 (samples: 141442)
INFO:root:Loss: 1.186 +/- 0.1499
INFO:root:Epoch no. 2 of 1000 (samples: 141442)
INFO:root:Loss: 0.7314 +/- 0.0373
...
INFO:root:Epoch no. 999 of 1000 (samples: 141442)
INFO:root:Loss: 0.0047 +/- 0.0002
INFO:root:Epoch no. 1000 of 1000 (samples: 141442)
INFO:root:Loss: 0.0047 +/- 0.0002
INFO:root:### MICRO (validation raw):
INFO:root:      -- left   >> mean: 388.3386, median: 3.0, hits@10: 78.94%
INFO:root:      -- right  >> mean: 384.2968, median: 3.0, hits@10: 77.72%
INFO:root:      -- global >> mean: 386.3177, median: 3.0, hits@10: 78.33%
INFO:root:### MICRO (validation filtered):
INFO:root:      -- left   >> mean: 375.7578, median: 2.0, hits@10: 92.08%
INFO:root:      -- right  >> mean: 371.3732, median: 2.0, hits@10: 91.82%
INFO:root:      -- global >> mean: 373.5655, median: 2.0, hits@10: 91.95%
INFO:root:### MICRO (test raw):
INFO:root:      -- left   >> mean: 287.0284, median: 3.0, hits@10: 78.4%
INFO:root:      -- right  >> mean: 293.8098, median: 3.0, hits@10: 79.52%
INFO:root:      -- global >> mean: 290.4191, median: 3.0, hits@10: 78.96%
INFO:root:### MICRO (test filtered):
INFO:root:      -- left   >> mean: 273.8508, median: 2.0, hits@10: 92.44%
INFO:root:      -- right  >> mean: 281.8194, median: 2.0, hits@10: 92.56%
INFO:root:      -- global >> mean: 277.8351, median: 2.0, hits@10: 92.5%
```

Please note that our results improve over those in [1]: AdaGrad has better convergence guarantees than Stochastic Gradient Descent (SGD), which was used in [1].

[1] Bordes, A. et al. - Translating Embeddings for Modeling Multi-relational Data - NIPS 2013

[2] Duchi, J. et al. - Adaptive Subgradient Methods for Online Learning and Stochastic Optimization - JMLR 2011
