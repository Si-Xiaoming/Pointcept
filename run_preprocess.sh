RAW_DIR= # Directory containing the raw data files
PROCESSED_DIR= # Directory to save the processed data files
NUM_WORKERS=1
DATASET_NAME=s3dis  # Name of the dataset to preprocess
GRID_SIZE=0.3
# activate the conda virtual environment
conda activate pointcept
# Run the preprocessing script
python pointcept/datasets/preprocessing/${DATASET_NAME}/preprocess_${DATASET_NAME}.py \
  --dataset_root ${RAW_DIR} \
  --output_root ${PROCESSED_DIR} \
  --raw_root ${RAW_DIR} \
  --num_workers ${NUM_WORKERS} \
  --grid_size ${GRID_SIZE}