# Data Augmentation using Pre-trained Transformer Models

### This code is originally released from amazon-research package (https://github.com/amazon-research/transformers-data-augmentation) 
### Since in the paper, we mentioned https://github.com/varinf/TransformersDataAugmentation url, we are providing a copy of the same code here. 

Code associated with the [Data Augmentation using Pre-trained Transformer Models](https://www.aclweb.org/anthology/2020.lifelongnlp-1.3.pdf) paper

Code contains implementation of the following data augmentation methods 
- EDA (Baseline)
- Backtranslation  (Baseline)
- CBERT (Baseline)
- BERT Prepend (Our paper)
- GPT-2 Prepend (Our paper)
- BART Prepend (Our paper)

## DataSets 

In paper, we use three datasets from following resources 
 - STSA-2 : [https://github.com/1024er/cbert_aug/tree/crayon/datasets/stsa.binary](https://github.com/1024er/cbert_aug/tree/crayon/datasets/stsa.binary)
 - TREC : [https://github.com/1024er/cbert_aug/tree/crayon/datasets/TREC](https://github.com/1024er/cbert_aug/tree/crayon/datasets/TREC)
 - SNIPS : [https://github.com/MiuLab/SlotGated-SLU/tree/master/data/snips](https://github.com/MiuLab/SlotGated-SLU/tree/master/data/snips)

### Low-data regime experiment setup  
Run `src/utils/download_and_prepare_datasets.sh` file to prepare all datsets.  
`download_and_prepare_datasets.sh` performs following steps
1. Download data from github 
2. Replace numeric labels with text for STSA-2 and TREC dataset
3. For a given dataset, creates 15 random splits of train and dev data.

## Dependencies 
 
To run this code, you need following dependencies 
- Pytorch 1.5
- fairseq 0.9 
- transformers 2.9 

## How to run 
To run data augmentation experiment for a given dataset, run bash script in `scripts` folder.
For example, to run data augmentation on `snips` dataset, 
 - run `scripts/bart_snips_lower.sh`  for BART experiment 
 - run `scripts/bert_snips_lower.sh` for rest of the data augmentation methods 


## How to cite 
```{bibtex}
@inproceedings{kumar-etal-2020-data,
    title = "Data Augmentation using Pre-trained Transformer Models",
    author = "Kumar, Varun  and
      Choudhary, Ashutosh  and
      Cho, Eunah",
    booktitle = "Proceedings of the 2nd Workshop on Life-long Learning for Spoken Language Systems",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.lifelongnlp-1.3",
    pages = "18--26",
}
```

## Contact

Please reachout to kuvrun@amazon.com for any questions related to this code. 

## License

This project is licensed under the Creative Common Attribution Non-Commercial 4.0 license.

   


