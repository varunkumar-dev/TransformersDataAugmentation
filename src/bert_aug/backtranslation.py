# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import csv
import logging
import argparse
import random

import os
import numpy as np
import torch
from fairseq.models.transformer import TransformerModel
from data_processors import get_task_processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="aug_data", type=str,
                        help="The output dir for augmented dataset")
    parser.add_argument("--task_name",default="subj",type=str,
                        help="The name of the task to train.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--sample_num', type=int, default=1,
                        help="sample number")
    parser.add_argument('--cache', default="fairseq_cache", type=str)
    parser.add_argument('--gpu', type=int, default=0,
                        help="gpu id")
    args = parser.parse_args()

    print(args)
    backtranslation_using_en_de_model(args)


def backtranslation_using_en_de_model(args):
    task_name = args.task_name
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    os.makedirs(args.output_dir, exist_ok=True)
    processor = get_task_processor(task_name, args.data_dir)
    # load train and dev data
    train_examples = processor.get_train_examples()

    # load the best model
    en_de_model = TransformerModel.from_pretrained(
        os.path.join(args.cache, "wmt19.en-de.joined-dict.single_model"),
        checkpoint_file="model.pt",
        tokenizer='moses',
        bpe='fastbpe'
    )

    de_en_model = TransformerModel.from_pretrained(
        os.path.join(args.cache, "wmt19.de-en.joined-dict.single_model"),
        checkpoint_file="model.pt",
        tokenizer='moses',
        bpe='fastbpe'
    )

    # en_de_model.to(device)
    # de_en_model.to(device)

    save_train_path = os.path.join(args.output_dir, "bt_aug.tsv")
    save_train_file = open(save_train_path, 'w')
    tsv_writer = csv.writer(save_train_file, delimiter='\t')
    for example in train_examples:
        text = example.text_a
        de_example = en_de_model.translate(text, remove_bpe=True)
        back_translated_example = de_en_model.translate(de_example, remove_bpe=True)
        tsv_writer.writerow([example.label, back_translated_example])


if __name__ == "__main__":
    main()