# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from collections import defaultdict
import argparse
import os
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_data_file(exp_id, source_file, target_file, num_examples=None, to_lower=False):
    random.seed(exp_id)
    target_category_data = defaultdict(list)

    with open(target_file, "w") as out_file, open(source_file, "r") as in_file:
        for line in in_file:
            if to_lower:
                line = line.lower()
            fields = line.strip().split("\t")
            if len(fields) == 2:
                category = fields[0]
                example = fields[1]
            else:
                raise ValueError("Unknown format. Expecting a two col tsv file")

            two_col_line = "\t".join([category, example])
            if num_examples is None:
                out_file.write(two_col_line)
                out_file.write("\n")
            else:
                target_category_data[category].append(two_col_line)

        if num_examples:
            # write num_seed utterances from target_category_data
            for cat, cat_data in target_category_data.items():
                if num_examples < len(cat_data):
                    seed_utterances = random.sample(cat_data, num_examples)
                else:
                    seed_utterances = cat_data
                for two_col_line in seed_utterances:
                    out_file.write(two_col_line)
                    out_file.write("\n")


def split_data(data_dir, num_train, num_dev, num_simulations, lower):
    all_training_data_file = os.path.join(data_dir, "train.tsv")
    dev_data_file = os.path.join(data_dir, "dev.tsv")
    test_data_file = os.path.join(data_dir, "test.tsv")

    for exp_id in range(num_simulations):
        exp_folder_path = os.path.join(data_dir, "exp_{}_{}".format(exp_id, num_train))
        if not os.path.exists(exp_folder_path):
            os.mkdir(exp_folder_path)
        else:
            raise ValueError("Directory {} already exists".format(exp_folder_path))

        # randomly select train data
        target_train_file = os.path.join(exp_folder_path, "train.tsv")
        process_data_file(exp_id, all_training_data_file, target_train_file,
                          num_examples=num_train, to_lower=lower)

        # randomly select dev data
        target_dev_file = os.path.join(exp_folder_path, "dev.tsv")
        process_data_file(exp_id, dev_data_file, target_dev_file,
                          num_examples=num_dev, to_lower=lower)

        # copy test file as it is
        target_test_file = os.path.join(exp_folder_path, "test.tsv")
        process_data_file(exp_id, test_data_file, target_test_file,
                          num_examples=None, to_lower=lower)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select N utterances from target category')
    # data category group
    parser.add_argument('-datadir', help='Data Dir', type=str, required=True)

    # data split parameters
    parser.add_argument('-num_train', help='Number of training examples to select', type=int,
                        required=True)
    parser.add_argument('-num_dev', help='Number of dev examples to select', type=int,
                        required=True)
    parser.add_argument('-sim', help='Number of simulations', type=int, default=15)

    # data pre-processing steps
    parser.add_argument('-lower', action='store_true', default=False)
    args = parser.parse_args()
    split_data(data_dir=args.datadir,
               num_train=args.num_train,
               num_dev=args.num_dev,
               num_simulations=args.sim,
               lower=args.lower)
