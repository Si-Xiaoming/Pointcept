import numpy as np
import sys
def convert_npy_to_ply(npy_pos_file, npy_label_file, ply_file):
    # Load the numpy arrays
    positions = np.load(npy_pos_file)
    labels = np.load(npy_label_file) # es

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

if __name__ == "__main__":
    npy_pos_file="/datasets/exp/default/result/ground_processed_coord.npy"
    npy_label_file="/datasets/exp/default/result/ground_processed_pred.npy"
    ply_file="/datasets/exp/default/result/ground_processed.ply"
    convert_npy_to_ply(npy_pos_file, npy_label_file, ply_file)