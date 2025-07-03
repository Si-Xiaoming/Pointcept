
CUDA_VISIBLE_DEVICES=3,4
NUM_GPU=1
CONFIG_PATH=configs/s3dis/semseg-pt-v3m1-1-rpe.py
SAVE_PATH=outputs/s3dis/semseg-pt-v3m1-1-rpe
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}


--config-file configs/s3dis/semseg-pt-v3m1-1-rpe.py --num-gpus 1 --options save_path=outputs/s3dis/semseg-pt-v3m1-1-rpe