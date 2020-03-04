# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Original Copyright Facebook, Inc. and its affiliates. Licensed under the MIT License as part of
# fairseq package.


import argparse
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe_utils import get_encoder


def get_labels(dataset_name, to_lower=True):
    """add your dataset here"""
    task_name = dataset_name.lower()
    if task_name == 'stsa':
        labels = ["Positive", "Negative"]
    elif task_name == 'trec':
        labels = ['Description', 'Entity', 'Abbreviation', 'Human', 'Location', 'Numeric']
    elif task_name == "snips":
        labels = ["PlayMusic", "GetWeather", "RateBook", "SearchScreeningEvent",
                "SearchCreativeWork", "AddToPlaylist", "BookRestaurant"]
    else:
        raise ValueError("unknown dataset {}".format(dataset_name))
    if to_lower:
        return [l.lower() for l in labels]
    else:
        return labels


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help='path to encoder.json',
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument(
        "--tsv",
        action="store_true",
        help="Is a TSV file. If true, will merge the columns",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Replace the labels with single BPE token",
    )
    parser.add_argument(
        "--dataset",
        default="sst2", type=str,
        help="Dataset. Used for filtering invalid utterances",
    )

    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)

        if args.decode:
            processed_lines = pool.imap(encoder.decode_lines, zip(*inputs), 100)
        else:
            processed_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, _lines) in enumerate(processed_lines, start=1):
            if filt == "PASS":
                for _line, output_h in zip(_lines, outputs):
                    print(_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        labels = get_labels(self.args.dataset)
        label_to_bpe_codes = {labels[i]: str(i+50265) for i in range(len(labels))}

        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]

            if self.args.tsv: # merge columns
                fields = line.split("\t")
                label = fields[0]
                text = fields[1]
                if self.args.label:
                    tokens = [label_to_bpe_codes[label]] + self.encode(text)
                else:
                    line = " ".join([label, text])
                    tokens = self.encode(line)
            else:
                tokens = self.encode(line)

            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        labels = get_labels(self.args.dataset)
        bpe_to_label_dict = {i + 50265: labels[i] for i in range(len(labels))}

        dec_lines = []
        for line in lines:
            if self.args.tsv: # write in tsv format
                if self.args.label:
                    tokens = line.strip().split()
                    utterance_text = self.decode(tokens[1:])
                    decoded_text = bpe_to_label_dict[tokens[0]] + "\t" + " ".join(utterance_text)
                else:
                    try:
                        tokens = map(int, line.strip().split())
                        decoded_text = self.decode(tokens)
                    except:
                        print(line)
                        continue

                    word_tokens = decoded_text.strip().split(" ")
                    if word_tokens[0] in labels:
                        decoded_text = word_tokens[0] + "\t" + " ".join(word_tokens[1:])
                    else:
                        print("Invalid utterance {}".format(word_tokens))
                        continue
            else:
                tokens = map(int, line.strip().split())
                decoded_text = self.decode(tokens)
            dec_lines.append(decoded_text)
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()

