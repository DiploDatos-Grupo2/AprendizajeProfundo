#!/usr/bin/env bash

set -ex

#if [ ! -d "./data/meli-challenge-2019/" ]
#then
#    mkdir -p ./data
#    echo >&2 "Downloading Meli Challenge Dataset"
#    curl -L https://cs.famaf.unc.edu.ar/\~ccardellino/resources/diplodatos/meli-challenge-2019.tar.bz2 -o ./data/meli-challenge-2019.tar.bz2
#    tar jxvf ./data/meli-challenge-2019.tar.bz2 -C ./data/
#fi
#
if [ ! -f "./data/SBW-vectors-300-min5.txt.gz" ]
then
    mkdir -p ./data
    echo >&2 "Downloading SBWCE"
    curl -L https://cs.famaf.unc.edu.ar/\~ccardellino/resources/diplodatos/SBW-vectors-300-min5.txt.gz -o ./data/SBW-vectors-300-min5.txt.gz
fi

# Be sure the correct nvcc is in the path with the correct pytorch installation
export CUDA_HOME=/opt/cuda/10.1
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

#python -m experiment.mlp \
#    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
#    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
#    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
#    --language spanish \
#    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
#    --embeddings-size 300 \
#    --hidden-layers 256 128 \
#    --dropout 0.3 \
#    --epochs 6

#python -m experiment.cnn1 \
#    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
#    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
#    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
#    --language spanish \
#    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
#    --embeddings-size 300 \
#    --hidden-layers 256 128 \
#    --dropout 0.5 \
#    --epochs 8

#python -m experiment.cnn3 \
#    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
#    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
#    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
#    --language spanish \
#    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
#    --embeddings-size 300 \
#    --hidden-layers 256 128 \
#    --dropout 0.5 \
#    --epochs 4  

#python -m experiment.cnn4 \
#    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
#    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
#    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
#    --language spanish \
#    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
#    --embeddings-size 300 \
#    --hidden-layers 256 128 \
#    --dropout 0.7 \
#    --epochs 8

python -m experiment.rnn \
    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
    --language spanish \
    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
    --embeddings-size 300 \
    --hidden-layers 256 128 \
    --dropout 0.1 \
    --epochs 4
#
#python -m experiment.cnn5 \
#    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
#    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
#    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
#    --language spanish \
#    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
#    --embeddings-size 300 \
#    --hidden-layers 256 128 \
#    --dropout 0.7 \
#    --epochs 8