#!/usr/bin/env bash

set -x

PARTITION=OpenDialogLab
JOB_NAME=MVT
GPUS=1
PY_ARGS=${@:1}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

srun -p $PARTITION \
    --gres=gpu:${GPUS} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    python train_model.py ${PY_ARGS}
