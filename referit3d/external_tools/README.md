- emd
- chamfer
```shell
pip install ninja 

cd chamfer_dist
python setup.py install
# srun -p OpenDialogLab --gres=gpu:1 python setup.py install 编译指令调用GPU.
# srun -p OpenDialogLab --gres=gpu:1 pip install KNN_CUDA-0.2-py3-none-any.whl
cd emd 
python setup.py install

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Pointnet
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/
pip install pointnet2_ops_lib/ .

pip install timm 
```