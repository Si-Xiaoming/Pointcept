#!/bin/bash

# Create and activate the Conda environment
#conda create -n pointcept python=3.8 -y
#conda activate pointcept

# Install necessary packages
conda install ninja -y
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# Install spconv (SparseUNet)
pip install spconv-cu113

# Install CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# Build PointOps
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch (replace "ARCH LIST" with your actual architecture)
# For example: "7.5 8.0" for RTX 3000 and A100
TORCH_CUDA_ARCH_LIST="7.5 8.0" python setup.py install
cd ../..

# Install Open3D (optional)
pip install open3d
