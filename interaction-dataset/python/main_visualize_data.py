#!/usr/bin/env python

try:
    import lanelet2

    use_lanelet2_lib = True
except ImportError:
    import warnings

    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)
    print("Using visualization without lanelet2.")
    use_lanelet2_lib = False
    from utils import map_vis_without_lanelet

import argparse
import os
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
import matplotlib as mpl



from utils import dataset_reader
from utils import dataset_types
from utils import map_vis_lanelet2
from utils import tracks_vis
from utils import dict_utils


def update_plot():
    global fig, timestamp, title_text, track_dictionary, patches_dict, text_dict, axes, pedestrian_dictionary
    # update text and tracks based on current timestamp
    assert (timestamp <= timestamp_max), "timestamp=%i" % timestamp
    assert (timestamp >= timestamp_min), "timestamp=%i" % timestamp
    assert (timestamp % dataset_types.DELTA_TIMESTAMP_MS == 0), "timestamp=%i" % timestamp
    percentage = (float(timestamp) / timestamp_max) * 100
    title_text.set_text("\nts = {} / {} ({:.2f}%)".format(timestamp, timestamp_max, percentage))
    tracks_vis.update_objects_plot(timestamp, patches_dict, text_dict, axes,
                                   track_dict=track_dictionary, pedest_dict=pedestrian_dictionary)
    fig.canvas.draw()


def start_playback():
    global timestamp, timestamp_min, timestamp_max, playback_stopped
    playback_stopped = False
    plt.ion()
    while timestamp < timestamp_max and not playback_stopped:
        timestamp += dataset_types.DELTA_TIMESTAMP_MS
        start_time = time.time()
        update_plot()
        end_time = time.time()
        diff_time = end_time - start_time
        plt.pause(max(0.001, dataset_types.DELTA_TIMESTAMP_MS / 1000. - diff_time))
    plt.ioff()


class FrameControlButton(object):
    def __init__(self, position, label):
        self.ax = plt.axes(position)
        self.label = label
        self.button = Button(self.ax, label)
        self.button.on_clicked(self.on_click)

    def on_click(self, event):
        global timestamp, timestamp_min, timestamp_max, playback_stopped

        if self.label == "play":
            if not playback_stopped:
                return
            else:
                start_playback()
                return
        playback_stopped = True
        if self.label == "<<":
            timestamp -= 10 * dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == "<":
            timestamp -= dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == ">":
            timestamp += dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == ">>":
            timestamp += 10 * dataset_types.DELTA_TIMESTAMP_MS
        timestamp = min(timestamp, timestamp_max)
        timestamp = max(timestamp, timestamp_min)
        update_plot()


if __name__ == "__main__":

    # provide data to be visualized
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                                                        "files)", nargs="?")
    parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", default=0, nargs="?")
    parser.add_argument("load_mode", type=str, help="Dataset to load (vehicle, pedestrian, or both)", default="both",
                        nargs="?")
    parser.add_argument("--start_timestamp", type=int, nargs="?")
    parser.add_argument("--lat_origin", type=float,
                        help="Latitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    parser.add_argument("--lon_origin", type=float,
                        help="Longitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    args = parser.parse_args()

    if args.scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if args.load_mode != "vehicle" and args.load_mode != "pedestrian" and args.load_mode != "both":
        raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")

    # check folders and files
    error_string = ""

    # root directory is one above main_visualize_data.py file
    # i.e. the root directory of this project
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tracks_dir = os.path.join(root_dir, "recorded_trackfiles")
    maps_dir = os.path.join(root_dir, "maps")

    lanelet_map_ending = ".osm"
    lanelet_map_file = os.path.join(maps_dir, args.scenario_name[:-4] + lanelet_map_ending) #[:-4] added remove _val from scenario_name for map

    scenario_dir = os.path.join(tracks_dir, args.scenario_name)

    track_file_name = os.path.join(
        scenario_dir,
        scenario_dir.split("/")[-1] + ".csv" # added, file name is scenario_dir[-1],  scenario_dir.split("/")[-1]
    )

    pedestrian_file_name = os.path.join(
        scenario_dir,
        "pedestrian_tracks_" + str(args.track_file_number).zfill(3) + ".csv"
    )

    if not os.path.isdir(tracks_dir):
        error_string += "Did not find track file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(maps_dir):
        error_string += "Did not find map file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(scenario_dir):
        error_string += "Did not find scenario directory \"" + scenario_dir + "\"\n"
    if not os.path.isfile(lanelet_map_file):
        error_string += "Did not find lanelet map file \"" + lanelet_map_file + "\"\n"
    if not os.path.isfile(track_file_name):
        error_string += "Did not find track file \"" + track_file_name + "\"\n"
    if not os.path.isfile(pedestrian_file_name):
        flag_ped = 0
    else:
        flag_ped = 1
    if error_string != "":
        error_string += "Type --help for help."
        raise IOError(error_string)

    # create a figure
    fig, axes = plt.subplots(1, 1)
    #fig.canvas.set_window_title("Interaction Dataset Visualization") #commented

    # load and draw the lanelet2 map, either with or without the lanelet2 library
    lat_origin = args.lat_origin  # origin is necessary to correctly project the lat lon values of the map to the local
    lon_origin = args.lon_origin  # coordinates in which the tracks are provided; defaulting to (0|0) for every scenario
    print("Loading map...")
    if use_lanelet2_lib:
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
        laneletmap = lanelet2.io.load(lanelet_map_file, projector)
        map_vis_lanelet2.draw_lanelet_map(laneletmap, axes)
    else:
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)

    # load the tracks
    print("Loading tracks...")
    track_dictionary = None
    pedestrian_dictionary = None
    if args.load_mode == 'both':
        track_dictionary = dataset_reader.read_tracks(track_file_name)
        if flag_ped:
            pedestrian_dictionary = dataset_reader.read_pedestrian(pedestrian_file_name)

    elif args.load_mode == 'vehicle':
        track_dictionary = dataset_reader.read_tracks(track_file_name)
    elif args.load_mode == 'pedestrian':
        pedestrian_dictionary = dataset_reader.read_pedestrian(pedestrian_file_name)

    timestamp_min = 1e9
    timestamp_max = 0

    if track_dictionary is not None:
        for key, track in dict_utils.get_item_iterator(track_dictionary):
            timestamp_min = min(timestamp_min, track.time_stamp_ms_first)
            timestamp_max = max(timestamp_max, track.time_stamp_ms_last)
    else:
        for key, track in dict_utils.get_item_iterator(pedestrian_dictionary):
            timestamp_min = min(timestamp_min, track.time_stamp_ms_first)
            timestamp_max = max(timestamp_max, track.time_stamp_ms_last)

    if args.start_timestamp is None:
        args.start_timestamp = timestamp_min

    button_pp = FrameControlButton([0.2, 0.05, 0.05, 0.05], '<<')  #added
    button_p = FrameControlButton([0.27, 0.05, 0.05, 0.05], '<')
    button_f = FrameControlButton([0.4, 0.05, 0.05, 0.05], '>')
    button_ff = FrameControlButton([0.47, 0.05, 0.05, 0.05], '>>')

    button_play = FrameControlButton([0.6, 0.05, 0.1, 0.05], 'play')
    button_pause = FrameControlButton([0.71, 0.05, 0.1, 0.05], 'pause')

    # storage for track visualization
    patches_dict = dict()
    text_dict = dict()

    # visualize tracks
    print("Plotting...")
    timestamp = args.start_timestamp
    title_text = fig.suptitle("")
    playback_stopped = True
    update_plot()  # added
    ### added ### ### added to draw spatial regions on map
    #tracks_vis.update_objects_plot(timestamp, patches_dict, text_dict, axes,
    #                               track_dict=track_dictionary, pedest_dict=pedestrian_dictionary)

    
    # on DR_CHN_Merging_ZS_val
    # center junction

    axes.add_patch(Rectangle((1016, 935),
                             66 , 20,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))


    plt.show()


'''
# on DR_CHN_Roundabout_LN_val
    # left junction
    axes.add_patch(Rectangle((924, 1000),  # lower left corner coord
                             14, 7,  # W,H
                             fc='none',
                             ec='r',  # clump
                             lw=2))

    axes.add_patch(Rectangle((939, 998),
                             24 , 8,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((963, 995),
                             14,13,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # right junction
    axes.add_patch(Rectangle((1054, 994),  # lower left corner coord
                             15, 8,  # W,H
                             fc='none',
                             ec='r',  # clump
                             lw=2))

    axes.add_patch(Rectangle((1038, 996),
                             17, 8,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1026, 994),
                             13, 14,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    ### added ###


# on DDR_DEU_Roundabout_OF_val
    # left junction
    axes.add_patch(Rectangle((952, 1015),  # lower left corner coord
                             15, 6,  # W,H
                             fc='none',
                             ec='r',  # clump
                             lw=2))

    axes.add_patch(Rectangle((969, 1008),
                             14 , 6,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((984, 999),
                             7,11,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # bottom junction
    axes.add_patch(Rectangle((1006, 965), # lower left corner coord
                             6, 11,  # W,H
                             fc='none',
                             ec='r', # clump
                             lw=2))

    axes.add_patch(Rectangle((1003, 977),
                             7, 10,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1001, 987),
                             9, 8,  # W,H
                             fc='none',
                             ec='b',   # unclump
                             lw=2))

    # right junction
    axes.add_patch(Rectangle((1028, 997),  # lower left corner coord
                             13, 7,  # W,H
                             fc='none',
                             ec='r',  # clump
                             lw=2))

    axes.add_patch(Rectangle((1014, 1000),
                             11, 5,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1006, 1004),
                             14, 11,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    ### added ###
    plt.show()


# on DR_USA_Roundabout_FT_val
    # top junction
    axes.add_patch(Rectangle((1037, 1026),  # lower left corner coord
                             7, 5,  # W,H
                             fc='none',
                             ec='r',  # clump
                             lw=2))

    axes.add_patch(Rectangle((1033, 1020),
                             7, 5,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1024, 1014),
                             9, 9,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # bottom junction
    axes.add_patch(Rectangle((1050, 977), # lower left corner coord
                             14, 8,  # W,H
                             fc='none',
                             ec='r', # clump
                             lw=2))

    axes.add_patch(Rectangle((1041, 982),
                             9, 7,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1031, 985),
                             11, 9,  # W,H
                             fc='none',
                             ec='b',   # unclump
                             lw=2))

    # left junction
    axes.add_patch(Rectangle((971, 981),  # lower left corner coord
                             15, 8,  # W,H
                             fc='none',
                             ec='r',  # clump
                             lw=2))

    axes.add_patch(Rectangle((986, 981),
                             10, 9,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((996, 983),
                             15, 13,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    ### added ###
    plt.show()


# on DR_USA_Roundabout_EP_Val

    # top junction
    axes.add_patch(Rectangle((942, 1029),  # lower left corner coord
                             20, 12,  # W,H
                             fc='none',
                             ec='r',  # clump
                             lw=2))

    axes.add_patch(Rectangle((960, 1023),
                             13, 12,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((970, 1012),
                             13, 13,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # bottom junction
    axes.add_patch(Rectangle((971, 976), # lower left corner coord
                             12, 10,  # W,H
                             fc='none',
                             ec='r', # clump
                             lw=2))

    axes.add_patch(Rectangle((975, 985),
                             11, 6,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((978, 992),
                             16, 8,  # W,H
                             fc='none',
                             ec='b',   # unclump
                             lw=2))

    # right junction
    axes.add_patch(Rectangle((1016, 1013),  # lower left corner coord
                             10, 8,  # W,H
                             fc='none',
                             ec='r',  # clump
                             lw=2))

    axes.add_patch(Rectangle((1009, 1013),
                             8, 9,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((996, 1011),
                             12, 12,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    ### added ###
    plt.show()

# on DR_USA_Intersection_EP0_Val
    # left junction
    axes.add_patch(Rectangle((967,982), # lower left corner coord
                           13,5,  # W,H
                           fc='none',
                           ec='r',  # clump
                           lw=2))

    axes.add_patch(Rectangle((983, 982),
                             10, 5,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((992, 981),
                             10, 9,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # top junction
    axes.add_patch(Rectangle((995, 1004),  # lower left corner coord
                             5, 10,  # W,H
                             fc='none',
                             ec='r',  # clump
                             lw=2))

    axes.add_patch(Rectangle((994, 995),
                             5, 6,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((992, 986),
                             10, 10,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # bottom junction
    axes.add_patch(Rectangle((1024, 962), # lower left corner coord
                             4, 10,  # W,H
                             fc='none',
                             ec='r', # clump
                             lw=2))

    axes.add_patch(Rectangle((1025, 972),
                             4, 7,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1023, 978),
                             10, 6,  # W,H
                             fc='none',
                             ec='b',   # unclump
                             lw=2))

    ### added ###
    plt.show()


# on DR_USA_Roundabout_SR_val
    # left junction
    axes.add_patch(Rectangle((935,1008), # lower left corner coord
                           13,6,  # W,H
                           fc='none',
                           ec='r',  # clump
                           lw=2))

    axes.add_patch(Rectangle((949, 1009),
                             17, 5,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((966, 1006),
                             16, 10,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # bottom junction
    axes.add_patch(Rectangle((987, 967), # lower left corner coord
                             6, 14,  # W,H
                             fc='none',
                             ec='r', # clump
                             lw=2))

    axes.add_patch(Rectangle((988, 980),
                             6, 11,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((989, 991),
                             15, 15,  # W,H
                             fc='none',
                             ec='b',   # unclump
                             lw=2))

    # right junction
    axes.add_patch(Rectangle((1033, 1022),  # lower left corner coord
                             14, 9,  # W,H
                             fc='none',
                             ec='r',  # clump
                             lw=2))

    axes.add_patch(Rectangle((1020, 1024),
                             13, 9,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1001, 1028),
                             15, 24,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))


    ### added ###
    plt.show()

# on DR_USA_Intersection_MA_val
    # left junction
    axes.add_patch(Rectangle((960,998), # lower left corner coord
                           22,9,  # W,H
                           fc='none',
                           ec='r',  # clump
                           lw=2))

    axes.add_patch(Rectangle((982, 998),
                             18, 9,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1000, 989),
                             17, 18,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # top junction
    axes.add_patch(Rectangle((1006, 1024), # lower left corner coord
                             11, 14,  # W,H
                             fc='none',
                             ec='r', # clump
                             lw=2))

    axes.add_patch(Rectangle((1004, 1011),
                             15, 13,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1001, 1004),
                             21, 10,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # bottom junction
    axes.add_patch(Rectangle((1027, 963), # lower left corner coord
                             11, 16,  # W,H
                             fc='none',
                             ec='r', # clump
                             lw=2))

    axes.add_patch(Rectangle((1025, 979),
                             12, 12,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1019, 991),
                             21, 15,  # W,H
                             fc='none',
                             ec='b',   # unclump
                             lw=2))

    # right junction
    axes.add_patch(Rectangle((1052, 1006), # lower left corner coord
                             19, 5,  # W,H
                             fc='none',
                             ec='r', # clump
                             lw=2))

    axes.add_patch(Rectangle((1038, 1006),
                             14, 5,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1024, 1006),
                             15, 6,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    ### added ###
    
    
    # on DR_USA_Intersection_GL_val
    # left junction
    axes.add_patch(Rectangle((942,993), # lower left corner coord
                           14,8,  # W,H
                           fc='none',
                           ec='r',  # clump
                           lw=2))

    axes.add_patch(Rectangle((956, 991),
                             18, 9,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((975, 985),
                             17, 10,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # top junction
    axes.add_patch(Rectangle((1021, 1017), # lower left corner coord
                             10, 14,  # W,H
                             fc='none',
                             ec='r', # clump
                             lw=2))

    axes.add_patch(Rectangle((1017, 1005),
                             13, 9,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((1011, 989),
                             16, 13,  # W,H
                             fc='none',
                             ec='b',  # unclump
                             lw=2))

    # bottom junction
    axes.add_patch(Rectangle((987, 953), # lower left corner coord
                             10, 10,  # W,H
                             fc='none',
                             ec='r', # clump
                             lw=2))

    axes.add_patch(Rectangle((984, 965),
                             14, 13,  # W,H
                             fc='none',
                             ec='y',  # neutral
                             lw=2))

    axes.add_patch(Rectangle((977, 984),
                             20, 13,  # W,H
                             fc='none',
                             ec='b',   # unclump
                             lw=2))


    ### added ###
    
'''
