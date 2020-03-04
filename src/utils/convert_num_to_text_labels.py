# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse

def get_label_dict(dataset_name):
    """
    Map numeric labels to the actual labels for STSA-Binary and TREC dataset

    Returns: Dict of {int: label} mapping

    """
    if dataset_name == "stsa":
        return {"0": "Negative", "1": "Positive"}
    elif dataset_name == "trec":
        label_list = ['Description', 'Entity', 'Abbreviation', 'Human', 'Location', 'Numeric']
        return {str(k): label_list[k] for k in range(len(label_list))}
    else:
        raise ValueError("Unknown dataset name")


def prepare_data(input_file, output_file, dataset_name):
    """
    Remove header line from dataset and change numeric label to text labels
    """
    line_count = 0
    label_dict = get_label_dict(dataset_name)
    with open(output_file, "w") as out_fp:
        with open(input_file, "r") as in_fp:
            for line in in_fp:
                if line_count == 0:
                    line_count += 1
                    continue
                fields = line.strip().split("\t")
                sentence = fields[0]
                label = fields[1]
                out_fp.write("\t".join([label_dict[label], sentence]))
                out_fp.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replace Numeric labels with Text labels')
    group = parser.add_argument_group(title="I/O params")
    group.add_argument('-i', type=str, help='Input file')
    group.add_argument('-o', type=str, help='Output file')
    group.add_argument('-d', type=str, help='DataSet name')

    args = parser.parse_args()
    prepare_data(args.i, args.o, args.d)