# Hyper: Latent Factor Models for Link Prediction in Knowledge Hypergraphs

This package implements several state-of-the-art latent factor models for link prediction in Knowledge Graphs such as the Translating Embeddings model, and their extensions for managing Knowledge (Hyper-)Graphs with super-dyadic relation types.

## Installation

This library is heavily based on Keras and Theano.

For installing a bleeding edge version of such libraries, run the following command:

```bash
$ sudo pip3 install --upgrade git+git://github.com/fchollet/keras.git git+git://github.com/Theano/Theano.git
```

Then, for installing this library as a user, run the following command:

```bash
$ python3 -uB setup.py install --user
```

## Running

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
