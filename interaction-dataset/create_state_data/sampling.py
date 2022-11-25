import pandas as pd
import numpy as np
import glob, os
from shutil import copy2

if __name__ == "__main__":
    path = '/home/mot_1/trajectory_prediction/dataset/interaction-dataset/interaction_dataset_state/test/'
    files = glob.glob(path+"*.txt")

    clump = [ file for file in files if file.split('/')[-1].split('_')[3]=="clump" ]
    unclump = [file for file in files if file.split('/')[-1].split('_')[3] == "unclump"]
    neutral = [file for file in files if file.split('/')[-1].split('_')[3] == "neutral"]

    #undersampling
    clump = clump[:17]
    unclump = unclump[:125]
    neutral = neutral[:125]

    for f in clump:
        copy2(f,'/home/mot_1/trajectory_prediction/dataset/interaction-dataset/interaction_dataset_state/sampled/test/')
    for f in unclump:
        copy2(f,'/home/mot_1/trajectory_prediction/dataset/interaction-dataset/interaction_dataset_state/sampled/test/')
    for f in neutral:
        copy2(f,'/home/mot_1/trajectory_prediction/dataset/interaction-dataset/interaction_dataset_state/sampled/test/')


'''
perform under sampling for class balancing
train data files count:
clump: 350 from 876
unclump: 350 from 2143
neutral: 350 from 6542
total: 1050

val data files count:
clump: 17 from 17
unclump: 125 from 416
neutral: 125 from 1276
total: 267
'''