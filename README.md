# Detecting False Positive and False Negative Overlaps in Assembly

Deep learning models that classify overlaps produced by assembly algorithms like [Raven](https://github.com/lukapozega/raven).

## Usage

`Data` directory contains examples of training data.

```
usage: main.py [-h] [--dataset DATASET] [--learning_rate] [--cuda] [--type TYPE] [--extended_cigar] [--fn]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  train dataset file
  --learning_rate    learning rate for Adam (default: 0.001)
  --cuda             use CUDA for model training
  --type TYPE        Model type: A, B or C
  --extended_cigar   Use extended cigar strings with X and = instead of M
  --fn               Classify false negatives
```

When using `extended_cigar` option, dataset must have `=` and `X` for matches, for example:

```
python3 main.py --dataset data/example.txt --fn --extended_cigar
```

whereas if `extended_cigar` flag is not passed, dataset must contain only `M` for matches:

```
python3 main.py --dataset data/exampleM.txt --fn
```


`fn` option classifies true positives, false positives and false negatives, while without the option model classifies only in true and false positive class.
