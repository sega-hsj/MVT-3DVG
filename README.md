# MVT-3DVG


## Installation
Please refer the installation of [referit3d](https://github.com/referit3d/referit3d)


## Training
Nr3D
```Console
    python referit3d/scripts/train_referit3d.py \
    -scannet-file PATH_OF_SCANNET_FILE \
    -referit3D-file PATH_OF_REFERIT3D_FILE \
    --bert-pretrain-path PATH_OF_BERT \
    --log-dir logs/MVT_nr3d \
    --n-workers 8 \
    --model 'referIt3DNet_transformer' \
    --unit-sphere-norm True \
    --batch-size 24 \
    --encoder-layer-num 3 \
    --decoder-layer-num 4 \
    --decoder-nhead-num 8 \
    --gpu "0" \
    --view_number 4 \
    --rotate_number 4 \
    --label-lang-sup True
```
