#!/usr/bin/env python

import csv
import operator
import pandas as pd
import numpy as np
from .dataset_types import MotionState, Track


class Key:
    track_id = "track_id"
    frame_id = "frame_id"
    time_stamp_ms = "timestamp_ms"
    agent_type = "agent_type"
    x = "x"
    y = "y"
    vx = "vx"
    vy = "vy"
    psi_rad = "psi_rad"
    length = "length"
    width = "width"


class KeyEnum:
    track_id = 0 + 1 # added 1 to every attribute since new case_id column is present in provided dataset
    frame_id = 1 + 1
    time_stamp_ms = 2 + 1
    agent_type = 3 + 1
    x = 4 + 1
    y = 5 + 1
    vx = 6 + 1
    vy = 7 + 1
    psi_rad = 8 + 1
    length = 9 + 1
    width = 10 + 1


def read_tracks(filename):

    # sorting based on track_id or else assertion error below
    '''csv_file = pandas.read_csv(filename)
    sorted_csv = csv_file.replace(np.nan,0)
    #sorted_csv = csv_file.sort_values(by=['track_id'])
    sorted_csv.to_csv("DR_USA_Intersection_MA_val.csv",index=False)
    exit()'''

    ### added modified code to read csv file with pandas library ###
    #with open(filename) as csv_file:
    #csv_reader = csv.reader(csv_file, delimiter=',') # commented
    csv_reader = pd.read_csv(filename, sep=',', index_col=False, header=0)
    csv_reader.sort_values('case_id', inplace=True)
    data_1 = csv_reader[csv_reader['case_id'] == 1].sort_values('track_id', inplace=False)
    data_2 = csv_reader[csv_reader['case_id'] == 2].sort_values('track_id', inplace=False)
    csv_reader =data_1
    #print(csv_reader)

    track_dict = dict()
    track_id = None

    for i, row in csv_reader.iterrows():

        '''if i == 0:
            # check first line with key names
            assert (row[KeyEnum.track_id] == Key.track_id)
            assert (row[KeyEnum.frame_id] == Key.frame_id)
            assert (row[KeyEnum.time_stamp_ms] == Key.time_stamp_ms)
            assert (row[KeyEnum.agent_type] == Key.agent_type)
            assert (row[KeyEnum.x] == Key.x)
            assert (row[KeyEnum.y] == Key.y)
            assert (row[KeyEnum.vx] == Key.vx)
            assert (row[KeyEnum.vy] == Key.vy)
            assert (row[KeyEnum.psi_rad] == Key.psi_rad)
            assert (row[KeyEnum.length] == Key.length)
            assert (row[KeyEnum.width] == Key.width)
            continue'''

        if int(row[KeyEnum.track_id]) != track_id:
            # new track
            track_id = int(row[KeyEnum.track_id])
            assert (track_id not in track_dict.keys()), \
                "Line %i: Track id %i already in dict, track file not sorted properly" % (i + 1, track_id)
            track = Track(track_id)
            track.agent_type = row[KeyEnum.agent_type]
            track.length = float(row[KeyEnum.length])
            track.width = float(row[KeyEnum.width])
            track.time_stamp_ms_first = int(row[KeyEnum.time_stamp_ms])
            track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
            track_dict[track_id] = track

        track = track_dict[track_id]
        track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
        ms = MotionState(int(row[KeyEnum.time_stamp_ms]))
        ms.x = float(row[KeyEnum.x])
        ms.y = float(row[KeyEnum.y])
        ms.vx = float(row[KeyEnum.vx])
        ms.vy = float(row[KeyEnum.vy])
        print("string:",row[KeyEnum.psi_rad])
        ms.psi_rad = float(row[KeyEnum.psi_rad])
        track.motion_states[ms.time_stamp_ms] = ms

    return track_dict


def read_pedestrian(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        track_dict = dict()
        track_id = None

        for i, row in enumerate(list(csv_reader)):

            if i == 0:
                # check first line with key names
                assert (row[KeyEnum.track_id] == Key.track_id)
                assert (row[KeyEnum.frame_id] == Key.frame_id)
                assert (row[KeyEnum.time_stamp_ms] == Key.time_stamp_ms)
                assert (row[KeyEnum.agent_type] == Key.agent_type)
                assert (row[KeyEnum.x] == Key.x)
                assert (row[KeyEnum.y] == Key.y)
                assert (row[KeyEnum.vx] == Key.vx)
                assert (row[KeyEnum.vy] == Key.vy)
                continue

            if row[KeyEnum.track_id] != track_id:
                # new track
                track_id = row[KeyEnum.track_id]
                assert (track_id not in track_dict.keys()), \
                    "Line %i: Track id %s already in dict, track file not sorted properly" % (i + 1, track_id)
                track = Track(track_id)
                track.agent_type = row[KeyEnum.agent_type]
                track.time_stamp_ms_first = int(row[KeyEnum.time_stamp_ms])
                track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
                track_dict[track_id] = track

            track = track_dict[track_id]
            track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
            ms = MotionState(int(row[KeyEnum.time_stamp_ms]))
            ms.x = float(row[KeyEnum.x])
            ms.y = float(row[KeyEnum.y])
            ms.vx = float(row[KeyEnum.vx])
            ms.vy = float(row[KeyEnum.vy])
            track.motion_states[ms.time_stamp_ms] = ms

        return track_dict
