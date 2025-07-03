S3DIS_DIR=/home/cm/share/SMS/dataset/S3DIS/raw/S3DIS
PROCESSED_S3DIS_DIR=/home/cm/share/SMS/dataset/S3DIS/PROCESS
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py 
--dataset_root ${S3DIS_DIR} 
--output_root ${PROCESSED_S3DIS_DIR} 
--raw_root ${RAW_S3DIS_DIR} --align_angle --parse_normal