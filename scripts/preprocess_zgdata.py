import pdal 
import os

def process_lidar_file(file_path, outname):
    # load las file
    pipeline = pdal.Pipeline()
    pipeline |= pdal.Reader.las(filename=file_path, nosrs=True)
    pipeline.execute()
    arrays = pipeline.arrays[0]
    metadata = pipeline.metadata['metadata']
    # convert cm to meter
    arrays['Z'] *= 0.01
    arrays['X'] *= 0.01
    arrays['Y'] *= 0.01

    pipeline = pdal.Pipeline(arrays=[arrays])
    las_kwargs = None # {'a_srs': metadata['readers.las']['comp_spatialreference']}
    pipeline |= pdal.Writer.las(filename=outname,
                                minor_version=4,
                                scale_x=0.001,
                                scale_y=0.001,
                                scale_z=0.001,
                                offset_x='auto',
                                offset_y='auto',
                                offset_z='auto',
                                ) # **las_kwargs
    pipeline.execute()

def process_batch_lidar_files(dirname, shuffix="_out.laz"):
    import os
    import glob

    files = glob.glob(os.path.join(dirname, '*.las'))
    for file in files:
        outname = os.path.splitext(file)[0] + shuffix
        process_lidar_file(file, outname)



def preprocess_zgdata(input_file, output_file):
    # read the las file
    pipeline = pdal.Pipeline()
    pipeline |= pdal.Reader.las(filename=input_file)
    pipeline |= pdal.Filter.stats(dimensions="Intensity,Red,Blue,Green")
    count = pipeline.execute()
    arrays = pipeline.arrays[0]

    # if point don't have color, remove it
    mask = (arrays['Red'] > 0) | (arrays['Green'] > 0) | (arrays['Blue'] > 0)
    arrays = arrays[mask]
    arrays['Z'] *= 0.01
    arrays['X'] *= 0.01
    arrays['Y'] *= 0.01
    pipeline = pdal.Pipeline(arrays=[arrays])
    las_kwargs = None # {'a_srs': metadata['readers.las']['comp_spatialreference']}
    pipeline |= pdal.Writer.las(filename=output_file,
                                minor_version=4,
                                scale_x=0.001,
                                scale_y=0.001,
                                scale_z=0.001,
                                offset_x='auto',
                                offset_y='auto',
                                offset_z='auto',
                                ) # **las_kwargs
    pipeline.execute()


if __name__ == "__main__":
    # dirname = r"D:\04-Datasets\test2\raw\train"
    # process_batch_lidar_files(dirname=dirname, shuffix="_processed.laz")

    dirname = "/datasets/internship"
    input_file = "merge_with_color.las"
    output_file = "zg_processed.laz"
    preprocess_zgdata(os.path.join(dirname, input_file),
                      os.path.join(dirname, output_file))