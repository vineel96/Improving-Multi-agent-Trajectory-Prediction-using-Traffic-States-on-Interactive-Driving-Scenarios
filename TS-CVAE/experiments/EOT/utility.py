import os
import numpy as np
import pandas as pd

if __name__=="__main__":
    df1 = pd.read_csv('/home/mot_1/trajectory_prediction/Trajectron-plus-plus/experiments/EOT/results_2fps_EOT_Combined/zara2_most_likely.csv')
    df2 = pd.read_csv('/home/mot_1/trajectory_prediction/Trajectron-plus-plus/experiments/EOT/results_2fps_EOT_Combined/zara2_best_of_20.csv')
    print("most likely:")
    print("ADE:", np.nanmean(df1['ade'][:-1]))
    print("FDE:",np.nanmean(df1['fde'][:-1]))
    print("best of 20:")
    print("ADE:", np.nanmean(df2['ade'][:-1]))
    print("FDE:", np.nanmean(df2['fde'][:-1]))
    print("KDE:", np.nanmean(df2['kde'][:-1]))