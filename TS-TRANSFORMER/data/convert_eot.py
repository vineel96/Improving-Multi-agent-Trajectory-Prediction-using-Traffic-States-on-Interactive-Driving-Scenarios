import os
import numpy as np
import glob
import sys
import subprocess
import argparse
sys.path.append(os.getcwd())


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="eth")
parser.add_argument('--raw_path', default="eth_data")
parser.add_argument('--out_path', default="datasets/eth_ucy")
args = parser.parse_args()

for mode in ['train', 'test', 'val']:
    raw_files = sorted(glob.glob(f'{args.raw_path}/{args.dataset}/{mode}/*.txt'))
    for raw_file in raw_files:
        raw_data = np.loadtxt(raw_file, delimiter=',')
        raw_data = raw_data[:,:4] # exclude last column of road user type
        raw_data[:, 0] = raw_data[:, 0] // 10
        raw_data[:, 0] -= raw_data[:, 0].min() # normalize frame id's

        # convert pixel to real world coords
        temp_split = (raw_file.split("/")[-1]).split('_')[0]
        if temp_split == "paldi":
            raw_data[:, 2]  = raw_data[:, 2] * (20 / 340)
            raw_data[:, 3] = raw_data[:, 3] * (20 / 340)
        elif temp_split == "nehru":
            raw_data[:, 2]  = raw_data[:, 2] * (20 / 315)
            raw_data[:, 3] = raw_data[:, 3] * (20 / 315)
        elif temp_split == "apmc":
            raw_data[:, 2]  = raw_data[:, 2] * (20 / 315)
            raw_data[:, 3] = raw_data[:, 3] * (20 / 315)

        new_data = np.ones([raw_data.shape[0], 17]) * -1.0
        new_data[:, 0] = raw_data[:, 0]
        new_data[:, 1] = raw_data[:, 1]
        new_data[:, [13, 15]] = raw_data[:, 2:4]
        new_data = new_data.astype(np.str_)
        new_data[:, 2] = 'Pedestrian'

        preprocess_dir_name = "EOT_split_preprocessed"
        out_file = f'{args.out_path}/{preprocess_dir_name}/{mode}/{os.path.basename(raw_file)}'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        np.savetxt(out_file, new_data, fmt='%s')
        print(out_file)