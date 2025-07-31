
CUDA_VISIBLE_DEVICES=0  # which gpu
NUM_GPU=1
CONFIG_PATH=configs/navarra/semseg-sonata-v1m1-0a-navarra-lin.py
SAVE_PATH=/datasets/navarra/semseg-sonata-v1m1-0a-navarra-lin
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

export PYTHONPATH=./   # Project path
# python -c "import torch; print(torch.cuda.is_available());"
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
