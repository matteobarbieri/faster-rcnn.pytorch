#!/bin/zsh

# Trains 

# Dataset and net used
DATASET=abbdoc
NET=res101

MAX_EPOCHS=10

CHECKPOINT_INTERVAL=2000
BS=2

# Arguments
SAVE_DIR=$1

python trainval_net.py \
	--dataset $DATASET \
	--net $NET \
	--bs $BS \
	--checkpoint_interval $CHECKPOINT_INTERVAL \
	--save_dir $SAVE_DIR \
	--epochs $MAX_EPOCHS \
	--use_tfb \
	--cuda
