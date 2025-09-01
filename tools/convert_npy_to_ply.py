import numpy as np
import sys
def convert_npy_to_ply(npy_pos_file, npy_label_file, ply_file):
    # Load the numpy arrays
    positions = np.load(npy_pos_file)
    labels = np.load(npy_label_file) # es

    # convert label [[1], [2]] to [1,2]
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.flatten()

    # Check if the shapes match
    if positions.shape[0] != labels.shape[0]:
        raise ValueError("Positions and labels must have the same number of points.")

    # Open the PLY file for writing
    with open(ply_file, 'w') as f:
        # Write the PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {positions.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar label\n")
        f.write("end_header\n")

        # Write the vertex data
        for pos, label in zip(positions, labels):
            f.write(f"{pos[0]} {pos[1]} {pos[2]} {label}\n")

def main_process(dir_path, data_name):
    npy_pos_file = f"{dir_path}/{data_name}_coord.npy"
    npy_label_file = f"{dir_path}/{data_name}_pred.npy"
    ply_file = f"{dir_path}/{data_name}.ply"
    convert_npy_to_ply(npy_pos_file, npy_label_file, ply_file)

if __name__ == "__main__":
    # dir_path = "/datasets/navarra-test2/processed/test/build3"
    # dir_path = "/datasets/exp/default_tcr/result"
    dir_path = "/datasets/exp/default/result"

    data_name = "build3_processed" # build3_processed   gr1_processed   ground_processed
    main_process(dir_path, data_name)




    