#!/bin/bash

# Trains 

if [ "$1" = "-h" ]; then
	echo -e "Usage:\n"
	echo -e "/do_training.sh NOTATION SAVE_DIR"
	exit 0
fi

# Dataset and net used
#DATASET=abbdoc
DATASET=$1
NET=res101

#MAX_EPOCHS=10
MAX_EPOCHS=6
#MAX_EPOCHS=30

CHECKPOINT_INTERVAL=2000
BS=5

# Arguments
SAVE_DIR=$2

TFB_LOGS=$3

if [ "$TFB_LOGS" = "" ]; then
	TFB_LOGS=/tmp/logs/$DATASET/`date "+%Y%m%d_%H%M"`
fi

#python trainval_net.py \
python test_val.py \
	--dataset $DATASET \
	--net $NET \
	--bs $BS \
	--checkpoint_interval $CHECKPOINT_INTERVAL \
	--save_dir $SAVE_DIR \
	--epochs $MAX_EPOCHS \
	--use_tfb \
	--tfb_logs $TFB_LOGS \
	--cuda
