#!/bin/bash

dataset_name=mesa
splits=train,validation,test
num_workers=4
model_path="sleepfm/checkpoints/model_base"

python generate_embeddings.py \
    --num_workers $num_workers \
    --batch_size 16 \
    --model_path $model_path \
    --dataset_name $dataset_name \
    --splits $splits
