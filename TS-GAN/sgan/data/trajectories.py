import logging
import os
import math

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, obs_traffic_state, pred_traffic_state) = zip(*data) #added

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    ## added ##
    obs_traffic_state = torch.cat(obs_traffic_state, dim=0).permute(2, 0, 1)
    pred_traffic_state = torch.cat(pred_traffic_state, dim=0).permute(2, 0, 1)
    ## added ##
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, obs_traffic_state, pred_traffic_state
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            line = line[:-1] # for EOT dataset exclude last value from annot file
            data.append(line)

    #normalizing frame-ids to [0,1,2...] for EOT dataset
    data = np.asarray(data)
    data[:,0] = data[:,0] // 10
    data[:,0] = data[:,0] - data[:,0].min()

    # normalizing x,y co-ords for EOT dataset
    #data[:, 2] = data[:, 2] / 4096
    #data[:, 3] = data[:, 3] / 2160

    # convert pixel to real world coords
    temp_split = (_path.split('/')[-1]).split('_')[0]
    if temp_split == "paldi":
        data[:, 2] = data[:, 2]  * (20 / 340)
        data[:, 3] = data[:, 3]  * (20 / 340)
    elif temp_split == "nehru":
        data[:, 2] = data[:, 2]  * (20 / 315)
        data[:, 3] = data[:, 3]  * (20 / 315)
    elif temp_split == "apmc":
        data[:, 2] = data[:, 2]  * (20 / 315)
        data[:, 3] = data[:, 3]  * (20 / 315)

    data[:, 2] = data[:, 2] - data[:, 2].mean()
    data[:, 3] = data[:, 3] - data[:, 3].mean()

    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t', mode=''
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = "," #delim
        delim = ","

        #all_files = os.listdir(self.data_dir)
        #all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        #all_files = [all_files[1]]
        all_files = []
        ### added ###
        train_files, val_files, test_files = [], [], []
        for file in os.listdir('/home/mot_1/trajectory_prediction/sgan-1/data/EOT_split/train/'): #/trajectory_prediction/
            train_files.append('/home/mot_1/trajectory_prediction/sgan-1/data/EOT_split/train/' + file)

        for file in os.listdir('/home/mot_1/trajectory_prediction/sgan-1/data/EOT_split/test/'):
            test_files.append('/home/mot_1/trajectory_prediction/sgan-1/data/EOT_split/test/' + file)

        for file in os.listdir('/home/mot_1/trajectory_prediction/sgan-1/data/EOT_split/val/'):
            val_files.append('/home/mot_1/trajectory_prediction/sgan-1/data/EOT_split/val/' + file)

        if mode == "train":
            all_files = train_files
        elif mode == "val":
            all_files = val_files
        elif mode == "test":
            all_files = test_files
        ### added ###

        traffic_state = [] #added traffic state
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:

            ### added ###
            if os.stat(path).st_size == 0:
                print("****EMPTY FILE*****")
                continue
            temp_split = (path.split('/')[-1]).split('_')[1]
            if temp_split == "clump":
                state_value = 0
            elif temp_split == "unclump":
                state_value = 1
            elif temp_split == "neutral":
                state_value = 2
            ### added ###

            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx   # idx acts as base pointer which is not required though
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    ### Some object tracklets will be missing in annot file as tracking algo is not robust, so remove those objects ###
                    if pad_end - pad_front != self.seq_len or curr_ped_seq.shape[0] != self.seq_len:  #added condition
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    #print("curr seq: ",curr_seq)
                    #print("curr ped seq: ",curr_ped_seq)
                    #print("length:",curr_ped_seq.shape)
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    ### added ###
                    num_peds_for_labels = curr_seq[:num_peds_considered].shape[0]
                    batch_traffic_state = np.zeros((num_peds_for_labels, 1, 20))
                    batch_traffic_state[:, 0, :] = state_value
                    traffic_state.append(batch_traffic_state)
                    ### added ###
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        traffic_state = np.concatenate(traffic_state, axis=0) #added
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.obs_traffic_state = torch.from_numpy(
            traffic_state[:, :, :self.obs_len]).type(torch.float)
        self.pred_traffic_state = torch.from_numpy(
            traffic_state[:, :, self.obs_len: ]).type(torch.float)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.obs_traffic_state[start:end, :], self.pred_traffic_state[start:end, :] #added
        ]
        return out
