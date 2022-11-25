import argparse
import os
import torch
import sys
import numpy as np
from scipy.stats import gaussian_kde

from attrdict import AttrDict
sys.path.append("../")

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1) # size: [269, 20]

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_

### added code to compute kde nll ###
def compute_kde_nll(predicted_samples, pred_traj_gt):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = pred_traj_gt.shape[1] # 12 ,  size: [269, 12, 2]
    num_batches = predicted_samples.shape[0]  # 269,  size: [269, 20, 12, 2]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_samples[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(pred_traj_gt[batch_num, timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan
    return -kde_ll

def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    kde_nll = []
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end, obs_traffic_state, pred_traffic_state) = batch #added

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)  # size: [12, 269, 2]

            predicted_samples = []
            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end, obs_traffic_state, pred_traffic_state  #added
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                predicted_samples.append(pred_traj_fake)
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ### added ###
            predicted_samples = torch.stack(predicted_samples,dim=0)  # size: [20, 12, 269, 2]
            predicted_samples = predicted_samples.permute(2,0,1,3) # size: [269, 20, 12, 2]
            pred_traj_gt = pred_traj_gt.permute(1,0,2) # size: [269, 12, 2]
            predicted_samples = predicted_samples.cpu().detach().numpy()
            pred_traj_gt = pred_traj_gt.cpu().detach().numpy()
            kde_nll.append(compute_kde_nll(predicted_samples,pred_traj_gt)) # compute kde nll
            ### added ###

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        print("KDE NLL: ", np.mean(kde_nll))
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        #path = get_dset_path(_args.dataset_name, args.dset_type)
        #path = '/home/mot_1/trajectory_prediction/sgan/scripts/datasets/zara2/test/'
        #path ='/home/mot_1/trajectory_prediction/sgan/data/EOT/unclump/test/'
        path = ''
        #print("History len:",_args.obs_len)
        #print("pred len:",_args.pred_len)
        _, loader = data_loader(_args, path, mode="test")
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.5f}, FDE: {:.5f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
