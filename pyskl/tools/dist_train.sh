#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=3,6

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export CUDA_LAUNCH_BLOCKING=1
set -x

CONFIG=$1
GPUS=$2

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}
