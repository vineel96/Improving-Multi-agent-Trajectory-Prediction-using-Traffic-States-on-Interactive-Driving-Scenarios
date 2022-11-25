import pandas as pd
import numpy as np
#pd.set_option("display.max_rows", None, "display.max_columns", None)

'''we treat val data as test data for evaluation purposes'''

if __name__=="__main__":
    full_data_path = '../recorded_trackfiles/train/DR_CHN_Merging_ZS2_train.csv'
    data = pd.read_csv(full_data_path, sep=',', index_col=False, header=0)  # change delimiter seperator
    data.sort_values('case_id', inplace=True)
    count = 0
    mode = "train"
    l_c_c, t_c_c, r_c_c, b_c_c =0, 0, 0, 0
    l_u_c, t_u_c, r_u_c, b_u_c =0, 0, 0, 0
    l_n_c, t_n_c, r_n_c, b_n_c =0, 0, 0, 0
    for case_id in pd.unique(data['case_id']):
        print("processing case id: ",case_id)
        l_c, t_c, r_c, b_c = [], [], [], []  # clump region at different junctions
        l_u, t_u, r_u, b_u = [], [], [], []  # unclump region at different junctions
        l_n, t_n, r_n, b_n = [], [], [], []  # neutral region at different junctions
        d = data[ data['case_id'] == case_id ]

        d = d[ d['agent_type'] == 'car']
        d.sort_values('frame_id', inplace=True)
        d = d[ ["frame_id", "track_id", "x", "y"] ] # filter only required columns
        if len(d) == 0:
            continue

        # neutral check
        l_n = d[ (d['x'] >= 1016) & (d['x'] <= 1016 + 66) & (d['y'] >= 935) & (d['y'] <= 935 + 20) ] # left junction

        # neutral
        # get boolean values where each object has atleast 20 positions
        l_n_check = np.array([len(pd.unique(l_n[l_n['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(l_n['track_id'])])

        if len(l_n) > 0 and len(pd.unique(l_n['track_id'])) > 1 and len(np.where(l_n_check == True)[0]) >=2:
            l_n.to_csv('../interaction_dataset_state/{}/Merging_ZS_left_neutral_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            l_n_c+=1

    print("l, t, r, b clump:",l_c_c, t_c_c, r_c_c, b_c_c )
    print("l, t, r, b unclump:", l_u_c, t_u_c, r_u_c, b_u_c)
    print("l, t, r, b neutral:", l_n_c, t_n_c, r_n_c, b_n_c)
    print("clump:",l_c_c+ t_c_c+ r_c_c+ b_c_c )
    print("unclump:",l_u_c+ t_u_c+ r_u_c+ b_u_c )
    print("neutral:",l_n_c+ t_n_c+ r_n_c+ b_n_c )

'''
val data files count:
l, t, r, b clump: 0 0 0 0
l, t, r, b unclump: 0 0 0 0
l, t, r, b neutral: 1032 0 0 0
clump: 0
unclump: 0
neutral: 1032
'''
