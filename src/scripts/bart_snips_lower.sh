#!/usr/bin/env bash

WARMUP_UPDATES=60
LR=1e-05                # Peak LR for polynomial LR scheduler.
SRC=~/PretrainedDataAugment/src
BART_PATH=~/bart.large
CACHE=~/CACHE
PREFIXSIZE=3
MAXEPOCH=30
TASK=snips

for NUMEXAMPLES in 10;
do
    for i in {0..14};
    do
    RAWDATADIR=~/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}
    DATABIN=$RAWDATADIR/jointdatabin

    splits=( train dev )
    for split in "${splits[@]}";
        do
        python $SRC/utils/bpe_encoder.py \
            --encoder-json $SRC/utils/gpt2_bpe/encoder.json \
            --vocab-bpe $SRC/utils/gpt2_bpe/vocab.bpe \
            --inputs $RAWDATADIR/${split}.tsv  \
            --outputs $RAWDATADIR/${split}_bpe.src \
            --workers 1 --keep-empty --tsv --dataset $TASK
        done

        fairseq-preprocess --user-dir=$SRC/bart_aug --only-source \
                    --task mask_s2s \
                    --trainpref $RAWDATADIR/train_bpe.src \
                    --validpref $RAWDATADIR/dev_bpe.src \
                    --destdir $DATABIN \
                    --srcdict $BART_PATH/dict.txt

        # Run data generation with different noise setting
        for mr in 40;
          do
            MRATIO=0.${mr}
            for MASKLEN in word span;
                do
                MODELDIR=$RAWDATADIR/bart_${MASKLEN}_mask_${MRATIO}_checkpoints
                mkdir $MODELDIR

                CUDA_VISIBLE_DEVICES=0 fairseq-train  $DATABIN/ \
                    --user-dir=$SRC/bart_aug \
                    --restore-file $BART_PATH/model.pt \
                    --arch bart_large \
                    --task mask_s2s \
                    --bpe gpt2 \
                    --gpt2_encoder_json $SRC/utils/gpt2_bpe/encoder.json \
                    --gpt2_vocab_bpe $SRC/utils/gpt2_bpe/vocab.bpe \
                    --layernorm-embedding \
                    --share-all-embeddings \
                    --save-dir $MODELDIR\
                    --seed $i \
                    --share-decoder-input-output-embed \
                    --reset-optimizer --reset-dataloader --reset-meters \
                    --required-batch-size-multiple 1 \
                    --max-tokens 2000 \
                    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
                    --dropout 0.1 --attention-dropout 0.1 \
                    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
                    --clip-norm 0.0 \
                    --lr-scheduler polynomial_decay --lr $LR \
                    --warmup-updates $WARMUP_UPDATES \
                    --replace-length 1 --mask-length $MASKLEN --mask $MRATIO --fp16 --update-freq 1 \
                    --max-epoch $MAXEPOCH --no-epoch-checkpoints > $MODELDIR/bart.log

               CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATABIN \
                        --user-dir=$SRC/bart_aug \
                        --task mask_s2s --tokens-to-keep $PREFIXSIZE \
                        --seed ${i} \
                        --bpe gpt2 \
                        --gpt2_encoder_json $SRC/utils/gpt2_bpe/encoder.json \
                        --gpt2_vocab_bpe $SRC/utils/gpt2_bpe/vocab.bpe \
                        --path $MODELDIR/checkpoint_best.pt \
                        --replace-length 1 --mask-length $MASKLEN --mask $MRATIO \
                        --batch-size 64 --beam 5 --lenpen 5 \
                        --no-repeat-ngram-size 2 \
                        --max-len-b 50 --prefix-size $PREFIXSIZE \
                        --gen-subset train > $MODELDIR/bart_l5_${PREFIXSIZE}.gen

                grep ^H $MODELDIR/bart_l5_${PREFIXSIZE}.gen | cut -f3 > $MODELDIR/bart_l5_gen_${PREFIXSIZE}.bpe
                rm $MODELDIR/checkpoint_last.pt
                python $SRC/utils/bpe_encoder.py \
                        --encoder-json $SRC/utils/gpt2_bpe/encoder.json \
                        --vocab-bpe $SRC/utils/gpt2_bpe/vocab.bpe \
                        --inputs $MODELDIR/bart_l5_gen_${PREFIXSIZE}.bpe \
                        --outputs $MODELDIR/bart_l5_gen_${PREFIXSIZE}.tsv --dataset $TASK \
                        --workers 1 --keep-empty --decode --tsv
            done
        done

        ########################
        ## BART Classifier
        ########################

         for mr in 40;
              do
                MRATIO=0.${mr}
                for MASKLEN in span word;
                    do
                     MODELDIR=$RAWDATADIR/bart_${MASKLEN}_mask_${MRATIO}_checkpoints

                    cat $RAWDATADIR/train.tsv $MODELDIR/bart_l5_gen_${PREFIXSIZE}.tsv > $MODELDIR/train.tsv
                    cp $RAWDATADIR/test.tsv $MODELDIR/test.tsv
                    cp $RAWDATADIR/dev.tsv $MODELDIR/dev.tsv
                    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $MODELDIR --seed ${i} --cache $CACHE > $RAWDATADIR/bert_bart_l5_${MASKLEN}_mask_${MRATIO}_prefix_${PREFIXSIZE}.log
                done
           done
    done
done
