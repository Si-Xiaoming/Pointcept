DATA_ROOT=/datasets/fine_tuning_data/
GRID_SIZE=1.0
python pointcept/datasets/preprocessing/navarra/navarra.py 
--dataset_root ${DATA_ROOT} 
--grid_size ${GRID_SIZE}
