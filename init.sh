#!/bin/bash
set -x

HOME=`pwd`

# compile custom operators
cd $HOME/referit3d/external_tools/chamfer_dist
# rm -rf build chamfer.egg-info
python setup.py install


cd $HOME/referit3d/external_tools/emd
# rm -rf build emd_ext.egg-info
python setup.py install


cd $HOME/referit3d/external_tools
pip install --upgrade KNN_CUDA-0.2-py3-none-any.whl


cd $HOME/referit3d/external_tools/pointnet2
# rm -rf build dist pointnet2.egg-info
python setup.py install

cd $HOME/referit3d/external_tools/Pointnet2_PyTorch
# rm -rf build pointnet2.egg-info
cd pointnet2_ops_lib/
# rm -rf build pointnet2_ops.egg-info
pip install .
