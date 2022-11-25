import sys
import os
import numpy as np
import pandas as pd
import dill
import shutil


sys.path.append("../../trajectron")
from environment import Environment, Scene, Node
from utils import maybe_makedirs
from environment import derivative_of

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
'''dt(seconds per frame) = 1 / fps; 1fps = 1Hz'''
'''if you need to know "real-world" velocity, you need the scaling factor (meters to pixels) as well.'''
dt = 0.5 #2fps #changed for EOT

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'traffic_state': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}


def augment_scene(scene, angle, state_value):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration','traffic_state'], ['x', 'y']]) #added state

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay,
                     ('traffic_state', 'x'): state_value,
                     ('traffic_state', 'y'): state_value  # added
                    }

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


nl = 0
l = 0
maybe_makedirs('../processed')
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration','traffic_state'], ['x', 'y']]) #added state variable

### added ###
'''train_files, val_files, test_files = [], [], []
for junc in ['apmc','nehru','paldi']:

    for traff in ['clump','unclump','neutral']:

        temp_p = 'raw/EOT_Combined_2fps/' + junc +"/" + traff
        files_present = os.listdir(temp_p)
        files_present.sort()
        total_n_files = len(files_present)
        num_train_files = int(total_n_files * 0.8)

        for t_ in files_present[:num_train_files]:
            train_files.append(temp_p + "/" + t_)
        for t_ in files_present[num_train_files:]:
            test_files.append(temp_p + "/" + t_)

val_files.append(test_files[0])
test_files = test_files[1:]
for pth in train_files:
    shutil.copy(pth, 'raw/EOT_split/train/' + pth.split('/')[-3] + "_" + pth.split('/')[-2] + "_" + pth.split('/')[-1])
for pth in test_files:
    shutil.copy(pth, 'raw/EOT_split/test/' + pth.split('/')[-3] + "_" + pth.split('/')[-2] + "_" + pth.split('/')[-1])
for pth in val_files:
    shutil.copy(pth, 'raw/EOT_split/val/' + pth.split('/')[-3] + "_" + pth.split('/')[-2] + "_" + pth.split('/')[-1])
exit()'''
### added ###

### added ###
train_files, val_files, test_files = [], [], []
for file in os.listdir('raw/EOT_split/train/'):
    train_files.append('raw/EOT_split/train/' + file)

for file in os.listdir('raw/EOT_split/test/'):
    test_files.append('raw/EOT_split/test/' + file)

for file in os.listdir('raw/EOT_split/val/'):
    val_files.append('raw/EOT_split/val/' + file)
### added ###


for desired_source in ['EOT_split_state']:#['EOT_2fps_10_state']:
    for data_class in ['train', 'val', 'test']:
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0 # should be different for car class in EOT dataset
        env.attention_radius = attention_radius

        scenes = []
        data_dict_path = os.path.join('../processed', '_'.join([desired_source, data_class]) + '.pkl')

        #for subdir, dirs, files in os.walk(os.path.join('raw', desired_source, "neutral", data_class)):
        for _ in range(1):
            if data_class == "train":
                files = train_files
            elif data_class == "val":
                files = val_files
            elif data_class == "test":
                files = test_files

            for file in files:
                if file.endswith('.txt'):
                    input_data_dict = dict()
                    full_data_path = file #os.path.join(subdir, file)
                    print('At', full_data_path)

                    ### added ###
                    if os.stat(full_data_path).st_size == 0:
                        print("****EMPTY FILE*****")
                        continue
                    ### added ###

                    ### added ###
                    # 0 - clump, 1 - unclump, 2 - neutral
                    state_value = -1
                    temp_split = (file.split("/")[-1]).split('_')[1]
                    if temp_split == "clump":
                        state_value = 0
                    elif temp_split == "unclump":
                        state_value = 1
                    elif temp_split == "neutral":
                        state_value = 2
                    ### added ###

                    data = pd.read_csv(full_data_path, sep=',', index_col=False, header=None) # change delimiter seperator
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y', 'road_user_type'] #added road user type
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

                    data['frame_id'] = data['frame_id'] // 10

                    data['frame_id'] -= data['frame_id'].min()

                    data['node_type'] = 'PEDESTRIAN'
                    data['node_id'] = data['track_id'].astype(str)
                    data.sort_values('frame_id', inplace=True)


                    #Normalize x,y coords  ( W x H: 4096 x 2160)
                    #data['pos_x'] = data['pos_x'] / 4096
                    #data['pos_y'] = data['pos_y'] / 2160

                    # convert pixel to real world coords
                    temp_split = (file.split("/")[-1]).split('_')[0]
                    if temp_split == "paldi":
                        data['pos_x'] = data['pos_x'] * (20 / 340)
                        data['pos_y'] = data['pos_y'] * (20 / 340)
                    elif temp_split == "nehru":
                        data['pos_x'] = data['pos_x'] * (20 / 315)
                        data['pos_y'] = data['pos_y'] * (20 / 315)
                    elif temp_split == "apmc":
                        data['pos_x'] = data['pos_x'] * (20 / 315)
                        data['pos_y'] = data['pos_y'] * (20 / 315)

                    # Mean Position
                    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                    max_timesteps = data['frame_id'].max()

                    scene = Scene(timesteps=max_timesteps+1, dt=dt, name=desired_source + "_" + data_class, aug_func=augment if data_class == 'train' else None)

                    for node_id in pd.unique(data['node_id']):

                        node_df = data[data['node_id'] == node_id]
                        #print(node_id)
                        #print(node_df)
                        #print(np.diff(node_df['frame_id']))
                        #assert np.all(np.diff(node_df['frame_id']) == 1)  # Track misses in EOT raise error( track might miss in some frames)

                        node_values = node_df[['pos_x', 'pos_y']].values

                        if node_values.shape[0] < 2:
                            continue

                        new_first_idx = node_df['frame_id'].iloc[0]

                        x = node_values[:, 0]
                        y = node_values[:, 1]
                        vx = derivative_of(x, scene.dt)
                        vy = derivative_of(y, scene.dt)
                        ax = derivative_of(vx, scene.dt)
                        ay = derivative_of(vy, scene.dt)

                        data_dict = {('position', 'x'): x,
                                     ('position', 'y'): y,
                                     ('velocity', 'x'): vx,
                                     ('velocity', 'y'): vy,
                                     ('acceleration', 'x'): ax,
                                     ('acceleration', 'y'): ay,
                                     ('traffic_state', 'x'): state_value,
                                     ('traffic_state', 'y'): state_value  #added
                                     }

                        node_data = pd.DataFrame(data_dict, columns=data_columns)
                        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                        node.first_timestep = new_first_idx

                        scene.nodes.append(node)
                    if data_class == 'train':
                        scene.augmented = list()
                        angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle, state_value)) #added state value

                    print(scene)
                    scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

print(f"Linear: {l}")
print(f"Non-Linear: {nl}")