## Text Classification Using a Convolutional Neural Network in Tensorflow

This branch of cnn-text-classsification-tf use [fassttext](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) pre-trained word vectors.

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy
- [gensim](https://radimrehurek.com/gensim/)

## Training

The steps for training CNN using fasttext are as follows.


```bash
# Download pre-trained fasttext word vectors
$ wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec

# Generate fasttext_vocab_en.dat, fasttext_embedding_en.npy
$ python util_fasttext.py

# Train a CNN using the pre-trained fasttext word vectors
$ python train.py --pre_trained

# The logs around step 1000 are as follows.
...
2017-07-08T13:12:27.329179: step 990, loss 0.178512, acc 0.953125
2017-07-08T13:12:28.902815: step 991, loss 0.133091, acc 0.984375
2017-07-08T13:12:30.473521: step 992, loss 0.148561, acc 0.984375
2017-07-08T13:12:32.041047: step 993, loss 0.21213, acc 0.90625
2017-07-08T13:12:33.617257: step 994, loss 0.230192, acc 0.9375
2017-07-08T13:12:35.223648: step 995, loss 0.222954, acc 0.9375
2017-07-08T13:12:36.822623: step 996, loss 0.161116, acc 0.96875
2017-07-08T13:12:38.437168: step 997, loss 0.224385, acc 0.921875
2017-07-08T13:12:40.073519: step 998, loss 0.258734, acc 0.921875
2017-07-08T13:12:41.649018: step 999, loss 0.207504, acc 0.953125
2017-07-08T13:12:43.215527: step 1000, loss 0.211571, acc 0.921875

Evaluation:
2017-07-08T13:12:44.823491: step 1000, loss 0.647888, acc 0.681
``` 

## Evaluating

```bash
# Create a symbolic link to the directory that have trained cnn model you want to evaluate.
# You can also copy all files to trained_cnn directory, but this need more disk spaces.
$ ln -s trained_cnn /path/to/directory/trained_cnn_model

# Evaluate on new data using the trained CNN with fasttext word vectors.
$ python eval.py --pre_trained
```

## Reference

* https://ireneli.eu/2017/01/17/tensorflow-07-word-embeddings-2-loading-pre-trained-vectors/
* https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27
