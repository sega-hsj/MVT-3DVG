# Multi-GPU branch for MVT-3DVG


## Build

```shell
cd MVT-3DVG

# compile custom operators
sh init.sh 
# To compile custom operators in slurm-multiple-GPUs system, please use the following command:
# srun -p OpenDialogLab --gres=gpu:1 sh init.sh

cd MVT-3DVG
pip install -e .

```

## Pretrained Point Transformer Model
If you would like to train MVT-3DVG with Point Transformer, download the pretrained [Point Transformer Model](https://drive.google.com/file/d/1Ms3K6fT5R85Td6ldMkFQkWAqX741oBdM/view?usp=share_link). 
- Update `point_tf_ckpt` in `referit3d/app/init_param.py` to point at the download point Transformer model.

## Referit3D Dataset
- Update other default parameters in `referit3d/app/init_param.py`. (ref [MVT-3DVG Commands](../../README.md))

## Training
- To train with multiple GPUs(slurm. multiple machines). Please modify the parameters (like `${PARTITION}, ${JOB_NAME}`) in the script as needed. 


```shell
cd referit3d/app

sh slurm_train.sh ${NUM_GPUS}
```

- To train with a single GPU:
```shell
sh single_gpu_train.sh
```

## Test and evaluate

- Download the [pretrained model](https://drive.google.com/file/d/1a0hvLW_lA489zd9ulb4Kn2rSx7Fjda-8/view?usp=share_link) and use the following command to test with multiple GPUs. 
```shell
# slurm: multiple GPUs
srun -p ${PARTITION} --gres=gpu:${NUM_GPUS} python train_model.py --mode evaluate --resume-path ${path: best_model.pth}

# The results should be:
Reference-Accuracy: 0.5715
Object-Clf-Accuracy: 0.6389
Text-Clf-Accuracy 0.9256
```