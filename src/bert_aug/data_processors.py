# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random
import os
import csv


def get_task_processor(task, data_dir):
    """
    A TSV processor for stsa, trec and snips dataset.
    """
    if task == 'stsa':
        return TSVDataProcessor(data_dir=data_dir, skip_header=False, label_col=0, text_col=1)
    elif task == 'trec':
        return TSVDataProcessor(data_dir=data_dir, skip_header=False, label_col=0, text_col=1)
    elif task == 'snips':
        return TSVDataProcessor(data_dir=data_dir, skip_header=False, label_col=0, text_col=1)
    else:
        raise ValueError('Unknown task')


def get_data(task, data_dir, data_seed=159):
    random.seed(data_seed)
    processor = get_task_processor(task, data_dir)

    examples = dict()

    examples['train'] = processor.get_train_examples()
    examples['dev'] = processor.get_dev_examples()
    examples['test'] = processor.get_test_examples()

    for key, value in examples.items():
        print('#{}: {}'.format(key, len(value)))
    return examples, processor.get_labels(task)


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

    def __getitem__(self, item):
        return [self.input_ids, self.input_mask,
                self.segment_ids, self.label_id][item]


class DatasetProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, task_name):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class TSVDataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, data_dir, skip_header, label_col, text_col):
        self.data_dir = data_dir
        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.tsv")), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "dev.tsv")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.tsv")), "test")

    def get_labels(self, task_name):
        """add your dataset here"""
        labels = set()
        with open(os.path.join(self.data_dir, "train.tsv"), "r") as in_file:
            for line in in_file:
                labels.add(line.split("\t")[self.label_col])
        return sorted(labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if self.skip_header and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.text_col]
            label = line[self.label_col]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
