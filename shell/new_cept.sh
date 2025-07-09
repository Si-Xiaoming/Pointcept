mamba create -n pointcept python=3.10 -y
conda activate pointcept

# (Optional) If no CUDA installed
mamba install nvidia/label/cuda-12.4.1::cuda conda-forge::cudnn conda-forge::gcc=13.2 conda-forge::gxx=13.2 -y



mamba install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
mamba install pytorch==2.5.0 torchvision==0.13.1 torchaudio==0.20.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
mamba install h5py pyyaml -c anaconda -y
mamba install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
mamba install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu124

# PPT (clip)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d