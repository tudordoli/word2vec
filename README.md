# Word2Vec from Scratch (Pure NumPy)

This project implements Word2Vec using only NumPy.

- No PyTorch
- No TensorFlow
- No automatic gradients

Everything is written manually:
- forward pass
- loss calculation
- gradient computation
- parameter updates

The model learns word vectors (lists of numbers) from text so that words that appear in similar contexts have similar vectors.

## What this code does

1. Reads a text file (Alice in Wonderland)
2. Builds training pairs of nearby words (skip-gram)
3. Trains embeddings with negative sampling + SGD
4. Prints a small nearest-neighbors sanity check at the end

## Project structure
.
├── word2vec_numpy.py
├── scripts/
│ └── download_gutenberg.py
├── data/ # created locally, not committed
├── requirements.txt
└── README.md

## Download dataset

python scripts/download_gutenberg.py

## Train

Minimal run:
python word2vec_numpy.py --corpus data/alice.txt

Example with hyperparameters:
python word2vec_numpy.py \
  --corpus data/alice.txt \
  --epochs 3 \
  --window 2 \
  --neg 5 \
  --embed-dim 100 \
  --lr 0.05 \
  --min-count 2 \
  --batch-size 256