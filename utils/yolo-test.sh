#!/bin/sh

set -ex

# DARKNET_SRC=$HOME/code/mirrors/github.com/pjreddie/darknet
# TEST_IMAGE=$DARKNET_SRC/data/dog.jpg

DARKNET_SRC=$HOME/code/repos/gitee.com/kuroicrow/darknet
# TEST_IMAGE=$DARKNET_SRC/sized.bmp
TEST_IMAGE=$DARKNET_SRC/sized.idx

MODEL_DIR=$HOME/var/models/yolo
LOG_DIR=$HOME/Desktop/log
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/crystalnet.log

time ./build/$(uname -s)/bin/yolo -m $MODEL_DIR -f $TEST_IMAGE 2> stderr.log | tee $LOG_FILE
echo "log saved to $LOG_FILE"
