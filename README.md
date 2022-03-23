# MVT-3DVG


## Installation
Please refer the installation of [referit3d](https://github.com/referit3d/referit3d)


## Training
* To train on either Nr3d or Sr3d dataset, use the following commands
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

* To train nr3d in joint with sr3d, add the following argument
```Console
    --augment-with-sr3d sr3d_dataset_file.csv
``` 

## Evaluation
* To evaluate on either Nr3d or Sr3d dataset, use the following commands
```Console
    cd referit3d/scripts/
    python train_referit3d.py --mode evaluate -scannet-file the_processed_scannet_file -referit3D-file dataset_file.csv --resume-path the_path_to_the_best_model.pth  --n-workers 4 --batch-size 64 
```
* To evaluate on joint trained model, add the following argument to the above command
```Console
    --augment-with-sr3d sr3d_dataset_file.csv
``` 
