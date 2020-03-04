# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel

from transformers import AdamW
from data_processors import get_task_processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPT2_MODEL = 'gpt2'
EOS_TOKEN = '<|endoftext|>'
SEP_TOKEN = '<SEP>'

STOP_TOKENS = [EOS_TOKEN, '<']

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, examples):
        self.examples = examples


def convert_examples_to_features(examples, block_size, tokenizer, seed=12345):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    text = ""
    for (ex_index, example) in enumerate(examples):
        if ex_index:
            text += " " + example.label + SEP_TOKEN + example.text_a + EOS_TOKEN
        else:
            text += example.label + SEP_TOKEN + example.text_a + EOS_TOKEN

    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    for i in range(0, len(tokenized_text) - block_size + 1,
                   block_size):  # Truncate in block of block_size
        features.append(InputFeatures(
            examples=tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size])))

    return features


def prepare_data(features):
    all_input_ids = torch.tensor([f.examples for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.examples for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_labels)
    return tensor_data


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="aug_data", type=str,
                        help="The output dir for augmented dataset")
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--block_size", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--cache', default="transformers_cache", type=str)
    parser.add_argument("--task_name", default="trec", type=str,
                        help="The name of the task to train.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=4e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
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
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--prefix", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0,
                        help="gpu id")
    parser.add_argument('--temp', type=float, default=1.0,
                        help="temperature")

    args = parser.parse_args()

    print(args)
    train_cmodgpt2_and_augment(args)


def compute_dev_loss(model, dev_dataloader):
    model.eval()
    sum_loss = 0.
    for step, batch in enumerate(dev_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'labels': batch[1]}

        outputs = model(**inputs)
        loss = outputs[0]
        sum_loss += loss.item()
    return sum_loss


def augment_train_data(model, tokenizer, train_examples, args):
    # load the best model
    best_model_path = os.path.join(args.output_dir, "best_cmodgpt2.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
    else:
        raise ValueError("Unable to find the saved model at {}".format(best_model_path))
    prefix_size = args.prefix
    save_train_path = os.path.join(args.output_dir, "cmodgpt2_aug_{}.tsv".format(prefix_size))
    save_train_file = open(save_train_path, 'w')

    tsv_writer = csv.writer(save_train_file, delimiter='\t')

    prefix_text = None
    for ex_index, example in enumerate(train_examples):
        model.eval()
        if prefix_size > 0:
            prefix_text = " ".join(example.text_a.split(' ')[:prefix_size])
            raw_text = example.label + SEP_TOKEN + prefix_text
        else:
            raw_text = example.label + SEP_TOKEN

        context_tokens = tokenizer.encode(raw_text, return_tensors='pt').to(device)
        out = model.generate(
            input_ids=context_tokens,
            max_length=args.max_seq_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=50256
        )

        out = out[:, len(context_tokens):].tolist()
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            eosn_index = 128
            for stop_token in STOP_TOKENS:
                idx = text.find(stop_token)
                if idx > 0:
                    eosn_index = min(eosn_index, idx)
            text = text[: eosn_index]
            text = text.replace("\n", " ").replace(EOS_TOKEN, ' ').strip()
            if prefix_size > 0:
                text = prefix_text + " " + text
            tsv_writer.writerow([example.label, text])


def train_cmodgpt2_and_augment(args):
    task_name = args.task_name
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    processor = get_task_processor(task_name, args.data_dir)
    #label_list = processor.get_labels(task_name)

    # load train and dev data
    train_examples = processor.get_train_examples()
    dev_examples = processor.get_dev_examples()

    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL,
                                              do_lower_case=True,
                                              cache_dir=args.cache)

    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL,
                                            cache_dir=args.cache)

    model.to(device)

    # train data
    train_features = convert_examples_to_features(train_examples,
                                                  args.block_size,
                                                  tokenizer, args.seed)
    train_data = prepare_data(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    # dev data
    dev_features = convert_examples_to_features(dev_examples,
                                                args.block_size,
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

    # Prepare optimizer and schedule (linear warmup and decay)
    t_total = num_train_steps
    no_decay = ['bias', 'LayerNorm.weight']
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

            inputs = {'input_ids': batch[0],
                      'labels': batch[1]}

            outputs = model(**inputs)
            loss = outputs[0]
            # loss = model(input_ids, segment_ids, input_mask, masked_ids)
            optimizer.zero_grad()
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            if (step + 1) % 50 == 0:
                print("avg_loss: {}".format(avg_loss / 50))
            # avg_loss = 0.

        # eval on dev after every epoch
        dev_loss = compute_dev_loss(model, dev_dataloader)
        print("Epoch {}, Dev loss {}".format(epoch, dev_loss))
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print("Saving model. Best dev so far {}".format(best_dev_loss))
            save_model_path = os.path.join(args.output_dir, 'best_cmodgpt2.pt')
            torch.save(model.state_dict(), save_model_path)

    # augment data using the best model
    augment_train_data(model, tokenizer, train_examples, args)


if __name__ == "__main__":
    main()