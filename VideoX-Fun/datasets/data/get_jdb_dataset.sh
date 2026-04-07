#!/bin/bash

# Get user input for data directory and dataset size

datadir=$1
dataset_size=$2 # small or all
num_proc=$3


# A. Download a small subset (~1%) of the dataset, if specified
if [ "$dataset_size" == "small" ]; then
    echo "Downloading ~1% of the dataset..."
    python download.py --datadir $datadir --max_image_size 512 --min_image_size 512 --valid_ids 11 --num_proc ${num_proc}
# Or download the entire dataset, if specified
elif [ "$dataset_size" == "all" ]; then
    echo "Downloading the full dataset... with processes " 
    echo ${num_proc}
    python ../prepare/jdb/download.py --datadir $datadir --max_image_size 512 --min_image_size 512 --num_proc ${num_proc}
else
    echo "Invalid dataset size option. Please use 'small' or 'all'."
    exit 1
fi

# B. Convert dataset to MDS format.
#python micro_diffusion/datasets/prepare/jdb/convert.py --images_dir "${datadir}/raw/train/imgs/" --captions_jsonl "${datadir}/raw/train/train_anno_realease_repath.jsonl" --local_mds_dir "${datadir}/mds/train/"
