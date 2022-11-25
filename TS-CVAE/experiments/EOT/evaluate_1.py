import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

### added ###
sys.path.insert(1,"/home/mot_1/trajectory_prediction/Trajectron-plus-plus-1/trajectron/") # change this path based on directory
from utils import prediction_output_to_trajectories
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib as mlp
mlp.use('Agg')
import numpy as np
import seaborn as sns
from scipy.spatial import ConvexHull
### added ###

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
args = parser.parse_args()

### added ###
device = 'cuda:0'
device = torch.device(device)
torch.cuda.set_device(device)
### added ###

### added ###
def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    colors = ['red', 'green', 'purple', 'darkgoldenrod', 'darkorange', 'peru', 'slategrey', 'hotpink', 'yellow',
              'cyan', 'teal', 'rosybrown', 'yellowgreen', 'chocolate', 'saddlebrown', 'crimson', 'dimgray', 'gainsboro',
              'tan', 'lightsteelblue']

    for idx, node in enumerate(histories_dict):  # loop through nodes present in present timestamp
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        # history.shape: (8, 2)
        # future.shape: (12, 2)
        # predictions.shape: (1, 20, 12, 2)
        # kde: False  , batch_num: 0

        # Get Mean Of 20 predicted trajectories
        pred_x_mean = np.mean(predictions[0, :, :, 0], axis=0)  # size: (20, 12), after mean:(12,)
        pred_y_mean = np.mean(predictions[0, :, :, 1], axis=0)  # size: (20, 12), after mean:(12,)

        ax.plot(history[:, 0], history[:, 1], 'x', color=colors[idx], alpha=1, label='History')

        ax.plot(pred_x_mean, pred_y_mean, 'o', color=colors[idx], alpha=1, label='Predicted Trajectory')

        ax.plot(future[:, 0], future[:, 1], '*', color='blue', alpha=1, label='Ground Truth')

        '''pred_points = predictions[0, :, :, :]
        pred_points = np.concatenate(pred_points, axis=0)

        hull = ConvexHull(pred_points)
        x_hull = np.append(pred_points[hull.vertices, 0],
                           pred_points[hull.vertices, 0][0])
        y_hull = np.append(pred_points[hull.vertices, 1],
                           pred_points[hull.vertices, 1][0])
        # plot shape
        plt.fill(x_hull, y_hull, alpha=0.1, c=colors[idx])'''

        for sample_num in range(prediction_dict[node].shape[1]): # 20 samples plotting if num_samples = 20

            sns.kdeplot(x = predictions[batch_num, sample_num, :, 0], y = predictions[batch_num, sample_num, :, 1],
                        ax=ax, shade=True, thresh=0.05,color=colors[idx], alpha=0.1, #levels = 5,
                        cut = 0.2)

            '''ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    'o', color=colors[idx], alpha=1, label='Predicted Trajectory')'''

    #plt.show()

def visualize_prediction( prediction_output_dict,
                         dt,
                         count,  # added
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    # predictions_dict keys are timestamps, each timestamp will have nodes info, each node shape: (1,20,length,2);length: HL or PH
    #assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return count
    ts_key = list(prediction_dict.keys())

    for ts in ts_key:  # loop through each timestamp
        fig, ax = plt.subplots()
        predict_dict = prediction_dict[ts]
        hist_dict = histories_dict[ts]
        future_dict = futures_dict[ts]

        if map is not None:
            ax.imshow(map.as_image(), origin='lower', alpha=0.5)
        plot_trajectories(ax, predict_dict, hist_dict, future_dict, *kwargs)
        plt.savefig('./plots/kde/plot_' + str(count) + '.png', bbox_inches='tight')
        plt.close()
        print("plotted graph no {}".format(count))
        count = count + 1

    return count


### added ###

def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, device)
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, device)

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])


    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']


    with torch.no_grad():
        ############### MOST LIKELY ###############
        '''eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        ade_mean_errors = np.array([]) #added
        fde_mean_errors = np.array([]) #added
        print("-- Evaluating GMM Grid Sampled (Most Likely)")
        for i, scene in enumerate(scenes):
            #print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            timesteps = np.arange(scene.timesteps)

            #print("before")
            #print(len(scene.nodes))
            predictions = eval_stg.predict(scene,
                                           timesteps,
                                           ph,
                                           num_samples=1,
                                           min_history_timesteps=7,
                                           min_future_timesteps=12,
                                           z_mode=False,
                                           gmm_mode=True,
                                           full_dist=True)  # This will trigger grid sampling


            s=0
            for k in predictions.keys():
                s = s + len(predictions[k].keys())

            batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                   scene.dt,
                                                                   max_hl=max_hl,
                                                                   ph=ph,
                                                                   node_type_enum=env.NodeType,
                                                                   map=None,
                                                                   prune_ph_to_future=True,
                                                                   kde=False)
            ### added ###
            #print("ADE Mean for Scene "+ str(i) + ": " + str(np.mean(batch_error_dict[args.node_type]['ade'])))
            #print("FDE Mean for Scene " + str(i) + ": " + str(np.mean(batch_error_dict[args.node_type]['fde'])))
            #print("No Of Nodes Evaluated for Scene " + str(i) + ": " + str(s))
            ade_mean_errors = np.hstack( (ade_mean_errors, np.nanmean(batch_error_dict[args.node_type]['ade']) ) )
            fde_mean_errors = np.hstack((fde_mean_errors, np.nanmean(batch_error_dict[args.node_type]['fde'])))

            ### added ###

            eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
            eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))

        ### added ###
        print("ADE Mean: ", np.nanmean(ade_mean_errors))
        print("FDE Mean: ", np.nanmean(fde_mean_errors))
        ade_mean_errors = np.hstack((ade_mean_errors, np.nanmean(ade_mean_errors)))
        fde_mean_errors = np.hstack((fde_mean_errors, np.nanmean(fde_mean_errors)))

        pd.DataFrame({'ade': ade_mean_errors, 'fde': fde_mean_errors , 'type': 'ml'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + 'most_likely.csv'))'''
        ### added ###
        '''print(np.mean(eval_fde_batch_errors))
        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_most_likely.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_most_likely.csv'))


        ############### MODE Z ###############
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        print("-- Evaluating Mode Z")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i+1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=2000,
                                               min_history_timesteps=7,
                                               min_future_timesteps=12,
                                               z_mode=True,
                                               full_dist=False)

                if not predictions:
                    continue

                batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=True)
                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'z_mode'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_z_mode.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'z_mode'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_z_mode.csv'))
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'z_mode'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_z_mode.csv'))'''


        ############### BEST OF 20 ###############
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        ade_mean_errors = np.array([])  # added
        fde_mean_errors = np.array([])  # added
        kde_mean_errors = np.array([])  # added
        count = 0 #added
        print("-- Evaluating best of 20")
        for i, scene in enumerate(scenes):
            #print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            s = 0
            ### added ###
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            ### added ###
            # removed for loop which loops over timestamp with interval 10
            timesteps =  np.arange(scene.timesteps)
            predictions = eval_stg.predict(scene,
                                           timesteps,
                                           ph,
                                           num_samples=20,
                                           min_history_timesteps=7,
                                           min_future_timesteps=12,
                                           z_mode=False,
                                           gmm_mode=False,
                                           full_dist=False)

            if not predictions:
                continue

            ### added ###
            # visualization
            #predictions : dict with keys->timestamps, value: nodes at that timestamp, each node shape: (1, 20, 12, 2)
            print("In Scene: ",i)
            count = visualize_prediction(predictions, scene.dt, count, max_hl=7, ph =12, map=None)
            ### added ###

            '''for k in predictions.keys():
                s = s + len(predictions[k].keys())

            batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                   scene.dt,
                                                                   max_hl=max_hl,
                                                                   ph=ph,
                                                                   node_type_enum=env.NodeType,
                                                                   map=None,
                                                                   best_of=True,
                                                                   prune_ph_to_future=True)

            eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
            eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
            eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))'''

            ### added ###
            #print("ADE Mean for Scene " + str(i) + ": " + str(np.mean(eval_ade_batch_errors)))
            #print("FDE Mean for Scene " + str(i) + ": " + str(np.mean(eval_fde_batch_errors)))
            #print("KDE Mean for Scene " + str(i) + ": " + str(np.mean(eval_kde_nll)))
            #print("No Of Nodes Evaluated for Scene " + str(i) + ": " + str(s))
            '''ade_mean_errors = np.hstack((ade_mean_errors, np.nanmean(eval_ade_batch_errors)))
            fde_mean_errors = np.hstack((fde_mean_errors, np.nanmean(eval_fde_batch_errors)))
            kde_mean_errors = np.hstack((kde_mean_errors, np.nanmean(eval_kde_nll)))'''
            ### added ###

        exit()
        ### added ###
        print("ADE Mean: ", np.nanmean(ade_mean_errors))
        print("FDE Mean: ", np.nanmean(fde_mean_errors))
        print("KDE Mean: ", np.nanmean(kde_mean_errors))
        ade_mean_errors = np.hstack((ade_mean_errors, np.nanmean(ade_mean_errors)))
        fde_mean_errors = np.hstack((fde_mean_errors, np.nanmean(fde_mean_errors)))
        kde_mean_errors = np.hstack((kde_mean_errors, np.nanmean(kde_mean_errors)))

        #pd.DataFrame({'ade': ade_mean_errors, 'fde': fde_mean_errors,'kde':kde_mean_errors, 'type': 'best_of'}
         #            ).to_csv(os.path.join(args.output_path, args.output_tag + 'best_of_20.csv'))
        ### added ###

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_best_of.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_best_of.csv'))
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_best_of.csv'))


        ############### FULL ###############
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        print("-- Evaluating Full")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=2000,
                                               min_history_timesteps=7,
                                               min_future_timesteps=12,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False)

                if not predictions:
                    continue

                batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=True)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'full'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_full.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'full'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_full.csv'))
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'full'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_full.csv'))
