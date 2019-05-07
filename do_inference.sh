#!/bin/bash

# Command line arguments
IMAGE_DIR=$1
NOTATION=$2

#SESSION=1
NET=res101
#DATASET=abbdoc
DATASET=$NOTATION

# This is actually useless
BS=2
#EPOCH=${3:=7}
#CHECKPOINT=$4
#CHECKPOINT=${4:=40000}

LOAD_DIR=$FASTERRCNN_MODELS_DIR

# XXX STUPID GPU PARALLELIZATION
if [ "$MOD_N" = "" ]; then
	MOD_N=-1
fi

if [ "$MOD_FILTER" = "" ]; then
	MOD_FILTER=-1
fi


#python demo_abbdoc.py \
python detect_symbols.py \
	--dataset $DATASET \
	--net $NET \
	--bs $BS \
	--load_dir $LOAD_DIR \
	--image_dir $IMAGE_DIR \
	--cuda \
	--mod_N $MOD_N \
	--mod_filter $MOD_FILTER \
	--cfg cfgs/res101.yml \
	--set ANCHOR_SCALES "[0.14, 0.35, 0.67, 1.19, 2.4]"
