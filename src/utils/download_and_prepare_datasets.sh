#!/usr/bin/env bash

mkdir datasets
NUMDEV=10
NUMEXP=15

# TREC dataset
mkdir -p datasets/trec
for split in train dev test;
  do
    wget -O datasets/trec/${split}.raw  https://raw.githubusercontent.com/1024er/cbert_aug/crayon/datasets/TREC/${split}.tsv
    python convert_num_to_text_labels.py -i datasets/trec/${split}.raw -o datasets/trec/${split}.tsv -d trec
    rm datasets/trec/${split}.raw
  done
python create_fsl_dataset.py -datadir datasets/trec -num_train 10 -num_dev $NUMDEV -sim $NUMEXP -lower


# STSA dataset
mkdir -p datasets/stsa
for split in train dev test;
  do
    wget -O datasets/stsa/${split}.raw  https://raw.githubusercontent.com/1024er/cbert_aug/crayon/datasets/stsa.binary/${split}.tsv
    python convert_num_to_text_labels.py -i datasets/stsa/${split}.raw -o datasets/stsa/${split}.tsv -d stsa
    rm datasets/stsa/${split}.raw
  done
python create_fsl_dataset.py -datadir datasets/stsa -num_train 10 -num_dev $NUMDEV -sim $NUMEXP -lower


# SNIPS dataset
mkdir -p datasets/snips
for split in train valid test;
  do
    wget -O datasets/snips/${split}.seq  https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/master/data/snips/${split}/seq.in
    wget -O datasets/snips/${split}.label  https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/master/data/snips/${split}/label
    paste -d'\t' datasets/snips/${split}.label datasets/snips/${split}.seq  > datasets/snips/${split}.tsv
    rm datasets/snips/${split}.label
    rm datasets/snips/${split}.seq
  done

mv datasets/snips/valid.tsv datasets/snips/dev.tsv
python create_fsl_dataset.py -datadir datasets/snips -num_train 10 -num_dev $NUMDEV -sim $NUMEXP -lower
