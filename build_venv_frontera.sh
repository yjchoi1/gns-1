#!/bin/bash

ml cuda/11.3
ml cudnn
ml nccl

module load intel/19.0.5
module load impi/19.0.5
module load phdf5
module load gcc/9.1.0
module load python3/3.8.2
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# start in frontera/venvs
python3 -m virtualenv venv-frontera3
source venv-frontera3/bin/activate

which python
python -m pip install --upgrade pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
python -m pip install -r ../gns-main/requirements.txt


# test env
# --------

echo 'which python -> venv'
which python

echo 'test_pytorch.py -> random tensor'
python ../gns-main/test/test_pytorch.py

echo 'test_pytorch_cuda_gpu.py -> True if GPU'
python ../gns-main/test/test_pytorch_cuda_gpu.py

echo 'test_torch_geometric.py -> no retun if import sucessful'
python ../gns-main/test/test_torch_geometric.py

# Clean up
# --------
#deactivate
#rm -r venv