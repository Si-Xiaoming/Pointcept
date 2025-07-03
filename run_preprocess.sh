RAW_DIR= # Directory containing the raw data files
PROCESSED_DIR= # Directory to save the processed data files

DATASET_NAME=s3dis  # Name of the dataset to preprocess

# activate the conda virtual environment
conda activate pointcept
# Run the preprocessing script
python pointcept/datasets/preprocessing/${DATASET_NAME}/preprocess_${DATASET_NAME}.py \
  --dataset_root ${RAW_DIR} \
  --output_root ${PROCESSED_DIR} \
  --raw_root ${RAW_DIR} \
  --align_angle \
  --parse_normal