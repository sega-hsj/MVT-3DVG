#!/usr/bin/env bash
###################################################################
# CMD:
#   1: train
#       nohup sh slurm_train.sh 8 > slurm_train_8.log 2>&1 &  
#       $ >>> [1] 125898
#       8 GPUs training, save log to slurm_train_8.log, PID is 125898
#   2: eval
#       nohup srun -p OpenDialogLab --gres=gpu:1 python train_model.py > slurm_eval_1.log 2>&1 &  
#   3: debug multi gpu training
#       # 注意: debug用到的参数已在下方写死: --point_trans --cls_head_finetune --use_fps
#       sh slurm_debug.sh 2
###################################################################

set -x

GPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

srun -p OpenDialogLab \
    --job-name=MVT \
    --gres=gpu:${GPUS} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    python -u train_model.py --dist_train True --tcp_port $PORT --point_trans --cls_head_finetune --use_fps ${PY_ARGS}


# 单独运行 Single GPU
# nohup srun -p OpenDialogLab --job-name=MVT --gres=gpu:1 --kill-on-bad-exit=1 --quotatype=reserved python train_model.py

# nohup
# > ~/projects/3d/MVT-3DVG/output/Logs/single_train.log 2>&1 &
# [1] 87292
# srun -p OpenDialogLab --job-name=MVT --gres=gpu:1 --kill-on-bad-exit=1 --quotatype=reserved python train_model.py 