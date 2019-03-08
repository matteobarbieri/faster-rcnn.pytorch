#!/bin/zsh

# Command line arguments
IMAGE_DIR=$1
NOTATION=$2

#SESSION=1
NET=res101
#DATASET=abbdoc
DATASET=$NOTATION
BS=2
#EPOCH=${3:=7}
#CHECKPOINT=$4
#CHECKPOINT=${4:=40000}

# TODO change these values!
#LOAD_DIR=${2:="/mnt/data/working_folder/ABB/models"}
LOAD_DIR="/mnt/data/working_folder/ABB/models"

#python demo_abbdoc.py \
python detect_symbols.py \
	--dataset $DATASET \
	--net $NET \
	--bs $BS \
	--load_dir $LOAD_DIR \
	--image_dir $IMAGE_DIR \
	--cuda \
	--set ANCHOR_SCALES "[4,8,16,32]"
