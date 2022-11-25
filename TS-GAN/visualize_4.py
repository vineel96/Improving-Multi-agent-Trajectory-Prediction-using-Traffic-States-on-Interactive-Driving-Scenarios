import argparse
import os
import torch
import numpy as np
from attrdict import AttrDict

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
import seaborn as sns

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

fake_tracks, real_tracks = [], []
cf_fakes_x, cf_fakes_y = [], []
cf_reals_x, cf_reals_y = [], []
fake_graph, real_graph = [], []

def gen_dot():
    for i in range(0, len(fake_tracks)):
        yield (real_tracks[i], fake_tracks[i])


def update_dot(newd):
    cf_reals_x.append(newd[0][0])
    cf_reals_y.append(newd[0][1])
    cf_fakes_x.append(newd[1][0])
    cf_fakes_y.append(newd[1][1])
    real_graph.set_data(cf_reals_x, cf_reals_y)
    fake_graph.set_data(cf_fakes_x, cf_fakes_y)
    return fake_graph, real_graph


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
    error = torch.stack(error, dim=1)
    sample_index = []

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        min_index = torch.argmin(_error)
        sample_index.append(min_index)
        _error = torch.min(_error)
        sum_ += _error
    return sum_, sample_index

def evaluate(args, loader, generator, num_samples):

    colors = ['red', 'green', 'purple', 'darkgoldenrod', 'darkorange', 'peru', 'slategrey', 'hotpink', 'yellow',
              'cyan', 'teal', 'rosybrown','yellowgreen', 'chocolate', 'saddlebrown', 'crimson', 'dimgray','gainsboro','tan',
              'lightsteelblue']
    with torch.no_grad():
        count = 0
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end, obs_traffic_state, pred_traffic_state) = batch

            ade, fde, predictions = [], [], []
            # Best Of 20 samples
            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end, obs_traffic_state, pred_traffic_state  # added
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                predictions.append(pred_traj_fake)  # save each samples predictions

            # visualize each sequence in the batch (i.e each sequence can have more than one agent)
            for index, seq in enumerate(seq_start_end):
                gt_track_data = np.array(pred_traj_gt[:, seq[0]:seq[1], :].data)
                obs_track_data = np.array(obs_traj[:, seq[0]:seq[1], :].data)

                predictions_seq = []
                for i in range(len(predictions)):
                    pred = predictions[i] # get ith prediction out of 20 predictions
                    pred = np.array(pred[:, seq[0]:seq[1], :].data) # slice predictions corresponding to present sequence
                    predictions_seq.append(pred)

                predictions_seq = np.array(predictions_seq) # size: (20, 12, 3, 2) ---> (samples, pred_horizon, agents, xy coords)
                # Get Mean Of 20 predicted trajectories
                pred_x_mean = np.mean(predictions_seq[:, :, :, 0],axis=0) # size: (20, 12, 3), after mean:(12,3)
                pred_y_mean = np.mean(predictions_seq[:, :, :, 1], axis=0) # size: (20, 12, 3), after mean:(12,3)

                #obs_track_data size: [8, num_peds, 2], #gt_track_data size: [12, num_peds, 2]
                cf, ax = plt.subplots()
                num_of_peds_in_seq = gt_track_data.shape[1]

                for idx in range(num_of_peds_in_seq): # assign markers colors seperately for each agent in scene

                    # added .flatten() as if 2d-array is given as input to plot, then labels appear twice in legend
                    hist_graph = ax.plot(obs_track_data[:, idx, 0], obs_track_data[:, idx, 1], 'x', color=colors[idx], alpha=1, label='History')
                    real_graph, = ax.plot(gt_track_data[:, idx, 0], gt_track_data[:, idx, 1], '*', color='blue', alpha=1, label='Ground Truth')
                    fake_graph, = ax.plot(pred_x_mean[:, idx], pred_y_mean[:, idx], 'o', color=colors[idx], alpha=1, label='Predicted Trajectory')

                    # create cloud shade of 20 trajectory predictions for each agent
                    prediction_agent = predictions_seq[:, :, idx, :] # get predictions of the current agent, size: (20, 12, 2)

                    for sample_num in range(20):
                        sns.kdeplot(data=prediction_agent[sample_num, :, 0], data2=prediction_agent[sample_num, :, 1],
                                ax=ax, shade=True, shade_lowest=False, color=colors[idx], alpha=0.1,  # levels = 5,
                                cut=0.2)

                    '''pred_points =  np.concatenate(prediction_agent,axis=0) # size: (240, 2)

                    hull = ConvexHull(pred_points)
                    x_hull = np.append(pred_points[hull.vertices, 0],
                                       pred_points[hull.vertices, 0][0])
                    y_hull = np.append(pred_points[hull.vertices, 1],
                                       pred_points[hull.vertices, 1][0])
                    # plot shape
                    plt.fill(x_hull, y_hull, alpha=0.1, c = colors[idx])'''

                #ax.legend()
                #plt.show()
                plt.savefig('./plots/with_pool_kde/plot_' + str(count) + '.png', bbox_inches='tight')
                plt.close()
                print("plotted graph no {}".format(count))
                count = count + 1


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
        path = ''
        _, loader = data_loader(_args, path, mode="test")
        evaluate(_args, loader, generator, args.num_samples)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
