# ========================================================================================
# Use ARG to define PyTorch, CUDA, and CUDNN versions
# ========================================================================================
ARG TORCH_VERSION=2.5.0
ARG CUDA_VERSION=12.4
ARG CUDNN_VERSION=9
ARG CUDA_VERSION_NO_DOT=124



FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

# ========================================================================================
# Set environment variables for CUDA
# ========================================================================================
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ========================================================================================
# Set proxy variables (if needed)
# ========================================================================================
ENV http_proxy="http://host.docker.internal:10809"
ENV https_proxy="http://host.docker.internal:10809"

# ========================================================================================
# Install APT packages
# ========================================================================================
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get -y install --no-install-recommends \
      git wget tmux vim zsh build-essential cmake ninja-build libopenblas-dev libsparsehash-dev \
      libgdal-dev gdal-bin python3-gdal openssh-server curl iputils-ping nano \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && export DEBIAN_FRONTEND=dialog

# ========================================================================================
# Configure SSH service
# ========================================================================================
RUN mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication no/' /etc/ssh/sshd_config
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

# ========================================================================================
# Install Conda packages
# ========================================================================================
RUN conda install -c conda-forge mamba
RUN mamba install h5py pyyaml -y
RUN mamba install tensorboard tensorboardx -y
RUN mamba install wandb yapf addict einops scipy -y
RUN mamba install plyfile termcolor matplotlib black open3d -y
RUN mamba install pdal python-pdal=3.2.3 -y

# ========================================================================================
# Install pip packages
# ========================================================================================
RUN pip install --upgrade pip 
RUN pip install timm numpy==1.26.4
RUN pip install torch_geometric torch_scatter torch_sparse torch_cluster  --index-url https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION_NO_DOT}.html 
RUN pip install spconv-cu124
RUN pip install ftfy regex tqdm omegaconf
RUN pip install argparse scikit-learn scipy

# ========================================================================================
# Install additional dependencies
# ========================================================================================
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
RUN \
    pip install git+https://github.com/octree-nn/ocnn-pytorch.git && \
    pip install git+https://github.com/openai/CLIP.git
RUN pip install -U git+https://github.com/microsoft/Swin3D.git -v
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation

# ========================================================================================
# Clone and install Pointcept
# ========================================================================================
RUN git clone https://github.com/Pointcept/Pointcept.git /Pointcept
WORKDIR /Pointcept
RUN pip install libs/pointops
RUN pip install libs/pointgroup_ops

WORKDIR /workspace