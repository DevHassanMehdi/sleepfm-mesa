#!/bin/bash

root_dir=data/mesa/edf
target_dir=data/mesa/hdf5

num_threads=4
num_files=-1
resample_rate=128

python preprocessing.py \
    --root_dir $root_dir \
    --target_dir $target_dir \
    --num_threads $num_threads \
    --num_files $num_files \
    --resample_rate $resample_rate
