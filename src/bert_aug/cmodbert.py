# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertForMaskedLM, BertOnlyMLMHead

from transformers import AdamW
from data_processors import get_task_processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_MODEL = 'bert-base-uncased'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, init_ids, input_ids, input_mask, masked_lm_labels):
        self.init_ids = init_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.masked_lm_labels = masked_lm_labels


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, seed=12345):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    # ----
    # dupe_factor = 5
    masked_lm_prob = 0.15
    max_predictions_per_seq = 20
    rng = random.Random(seed)


    for (ex_index, example) in enumerate(examples):
        modified_example = example.label + " " + example.text_a
        tokens_a = tokenizer.tokenize(modified_example)
        # Account for [CLS] and [SEP] and label with "- 3"
        if len(tokens_a) > max_seq_length - 3:
            tokens_a = tokens_a[0:(max_seq_length - 3)]

        # take care of prepending the class label in this code
        tokens = []
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")
        masked_lm_labels = [-100] * max_seq_length

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            # making sure that masking of # prepended label is avoided
            if token == "[CLS]" or token == "[SEP]" or (token in label_list and i == 1):
                continue
            cand_indexes.append(i)

        rng.shuffle(cand_indexes)
        len_cand = len(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms_pos = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms_pos) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = tokens[cand_indexes[rng.randint(0, len_cand - 1)]]

            masked_lm_labels[index] = tokenizer.convert_tokens_to_ids([tokens[index]])[0]
            output_tokens[index] = masked_token
            masked_lms_pos.append(index)

        init_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            init_ids.append(0)
            input_ids.append(0)
            input_mask.append(0)

        assert len(init_ids) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("init_ids: %s" % " ".join([str(x) for x in init_ids]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))

        features.append(
            InputFeatures(init_ids=init_ids,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          masked_lm_labels=masked_lm_labels))
    return features


def prepare_data(features):
    all_init_ids = torch.tensor([f.init_ids for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in features],
                                        dtype=torch.long)
    tensor_data = TensorDataset(all_init_ids, all_input_ids, all_input_mask, all_masked_lm_labels)
    return tensor_data


def rev_wordpiece(str):
    #print(str)
    if len(str) > 1:
        for i in range(len(str)-1, 0, -1):
            if str[i] == '[PAD]':
                str.remove(str[i])
            elif len(str[i]) > 1 and str[i][0]=='#' and str[i][1]=='#':
                str[i-1] += str[i][2:]
                str.remove(str[i])
    return " ".join(str[2:-1])


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="aug_data", type=str,
                        help="The output dir for augmented dataset")
    parser.add_argument("--task_name",default="subj",type=str,
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    # parser.add_argument("--do_lower_case", default=False, action='store_true',
    #                     help="Set this flag if you are using an uncased model.")
    parser.add_argument('--cache', default="transformers_cache", type=str)
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=4e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--sample_num', type=int, default=1,
                        help="sample number")
    parser.add_argument('--sample_ratio', type=int, default=7,
                        help="sample ratio")
    parser.add_argument('--gpu', type=int, default=0,
                        help="gpu id")
    parser.add_argument('--temp', type=float, default=1.0,
                        help="temperature")

    args = parser.parse_args()

    print(args)
    train_cmodbert_and_augment(args)


def compute_dev_loss(model, dev_dataloader):
    model.eval()
    sum_loss = 0.
    for step, batch in enumerate(dev_dataloader):
        batch = tuple(t.to(device) for t in batch)
        _, input_ids, input_mask, masked_ids = batch
        inputs = {'input_ids': batch[1],
                  'attention_mask': batch[2],
                  'masked_lm_labels': batch[3]}

        outputs = model(**inputs)
        loss = outputs[0]
        sum_loss += loss.item()
    return sum_loss


def augment_train_data(model, tokenizer, train_data, label_list, args):
    # load the best model

    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size)
    best_model_path = os.path.join(args.output_dir, "best_cmodbert.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        raise ValueError("Unable to find the saved model at {}".format(best_model_path))

    save_train_path = os.path.join(args.output_dir, "cmodbert_aug.tsv")
    save_train_file = open(save_train_path, 'w')

    MASK_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    tsv_writer = csv.writer(save_train_file, delimiter='\t')

    for step, batch in enumerate(train_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        init_ids, _, input_mask, _ = batch
        input_lens = [sum(mask).item() for mask in input_mask]
        masked_idx = np.squeeze(
            [np.random.randint(2, l, max((l-2) // args.sample_ratio, 1)) for l in input_lens])
        for ids, idx in zip(init_ids, masked_idx):
            ids[idx] = MASK_id

        inputs = {'input_ids': init_ids,
                  'attention_mask': input_mask}

        outputs = model(**inputs)
        predictions = outputs[0]  # model(init_ids, segment_ids, input_mask)
        predictions = F.softmax(predictions / args.temp, dim=2)

        for ids, idx, preds in zip(init_ids, masked_idx, predictions):
            preds = torch.multinomial(preds, args.sample_num, replacement=True)[idx]
            if len(preds.size()) == 2:
                preds = torch.transpose(preds, 0, 1)
            for pred in preds:
                ids[idx] = pred
                new_str = tokenizer.convert_ids_to_tokens(ids.cpu().numpy())
                label = new_str[1]
                new_str = rev_wordpiece(new_str)
                tsv_writer.writerow([label, new_str])


def train_cmodbert_and_augment(args):
    task_name = args.task_name
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    processor = get_task_processor(task_name, args.data_dir)
    label_list = processor.get_labels(task_name)

    # load train and dev data
    train_examples = processor.get_train_examples()
    dev_examples = processor.get_dev_examples()

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL,
                                              do_lower_case=True,
                                              cache_dir=args.cache)

    model = BertForMaskedLM.from_pretrained(BERT_MODEL,
                                            cache_dir=args.cache)

    tokenizer.add_tokens(label_list)
    # Adding embeddings such that they are randomly initialized, however, follow instructions about initializing them
    # intelligently
    model.resize_token_embeddings(len(tokenizer))
    model.cls = BertOnlyMLMHead(model.config)

    model.to(device)

    # train data
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  args.max_seq_length,
                                                  tokenizer, args.seed)
    train_data = prepare_data(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size)


    # dev data
    dev_features = convert_examples_to_features(dev_examples, label_list,
                                                  args.max_seq_length,
                                                  tokenizer, args.seed)
    dev_data = prepare_data(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler,
                                  batch_size=args.train_batch_size)

    num_train_steps = int(len(train_features) / args.train_batch_size * args.num_train_epochs)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    # Prepare optimizer
    t_total = num_train_steps
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    best_dev_loss = float('inf')
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        avg_loss = 0.
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            _, input_ids, input_mask, masked_ids = batch
            inputs = {'input_ids': batch[1],
                      'attention_mask': batch[2],
                      'masked_lm_labels': batch[3]}

            outputs = model(**inputs)
            loss = outputs[0]
            #loss = model(input_ids, segment_ids, input_mask, masked_ids)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            if (step + 1) % 50 == 0:
                print("avg_loss: {}".format(avg_loss / 50))
            avg_loss = 0.

        # eval on dev after every epoch
        dev_loss = compute_dev_loss(model, dev_dataloader)
        print("Epoch {}, Dev loss {}".format(epoch, dev_loss))
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print("Saving model. Best dev so far {}".format(best_dev_loss))
            save_model_path = os.path.join(args.output_dir, 'best_cmodbert.pt')
            torch.save(model.state_dict(), save_model_path)

    # augment data using the best model
    augment_train_data(model, tokenizer, train_data, label_list, args)


if __name__ == "__main__":
    main()