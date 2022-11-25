import pandas as pd
import numpy as np
#pd.set_option("display.max_rows", None, "display.max_columns", None)

'''we treat val data as test data for evaluation purposes'''

if __name__=="__main__":
    full_data_path = '../recorded_trackfiles/train/DR_USA_Roundabout_SR_train.csv'
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

        #clump check
        l_c = d[ (d['x'] >= 935) & (d['x'] <= 935 + 13) & (d['y'] >= 1008) & (d['y'] <= 1008 + 6)] # left junction
        #t_c = d[ (d['x'] >= 1006) & (d['x'] <= 1006 + 11 ) & (d['y'] >= 1024) & (d['y'] <= 1024 + 14) ] # top   junction
        r_c = d[ (d['x'] >= 1033) & (d['x'] <= 1033 + 14 ) & (d['y'] >= 1022) & (d['y'] <= 1022 + 9) ] # right   junction
        b_c = d[ (d['x'] >= 987) & (d['x'] <= 987 + 6 ) & (d['y'] >= 967) & (d['y'] <= 967 + 14) ] # bottom   junction
        # unclump check
        l_u = d[ (d['x'] >= 966) & (d['x'] <= 966 + 16) & (d['y'] >= 1006) & (d['y'] <= 1006 + 10)] # left junction
        #t_u = d[ (d['x'] >= 1001) & (d['x'] <= 1001 + 21) & (d['y'] >= 1004) & (d['y'] <= 1004 + 10) ] # top   junction
        r_u = d[ (d['x'] >= 1001) & (d['x'] <= 1001 + 15) & (d['y'] >= 1028) & (d['y'] <= 1028 + 24)] # right   junction
        b_u = d[ (d['x'] >= 989) & (d['x'] <= 989 + 15) & (d['y'] >= 991) & (d['y'] <= 991 + 15) ] # bottom   junction
        # neutral check
        l_n = d[ (d['x'] >= 949) & (d['x'] <= 949 + 17) & (d['y'] >= 1009) & (d['y'] <= 1009 + 5) ] # left junction
        #t_n = d[ (d['x'] >= 1004) & (d['x'] <= 1004 + 15) & (d['y'] >= 1011) & (d['y'] <= 1011 + 13) ] # top   junction
        r_n = d[ (d['x'] >= 1020) & (d['x'] <= 1020 + 13) & (d['y'] >= 1024) & (d['y'] <= 1024 + 9) ] # right   junction
        b_n = d[ (d['x'] >= 988) & (d['x'] <= 988 + 6) & (d['y'] >= 980) & (d['y'] <= 980 + 11) ] # bottom   junction

        # clump
        # get boolean values where each object has atleast 20 positions
        l_c_check = np.array([ len(pd.unique(l_c[ l_c['track_id'] == id ]['frame_id']))>=20 for id in pd.unique(l_c['track_id']) ])
        #t_c_check = np.array([len(pd.unique(t_c[t_c['track_id'] == id]['frame_id'])) >= 20 for id in pd.unique(t_c['track_id'])])
        r_c_check = np.array([len(pd.unique(r_c[r_c['track_id'] == id]['frame_id'])) >= 20 for id in pd.unique(r_c['track_id'])])
        b_c_check = np.array([len(pd.unique(b_c[b_c['track_id'] == id]['frame_id'])) >= 20 for id in pd.unique(b_c['track_id'])])

        if  len(l_c) > 0 and len(pd.unique(l_c['track_id'])) > 1 and len(np.where(l_c_check == True)[0]) >=2:
            l_c.to_csv('../interaction_dataset_state/{}/Roundabout_SR_left_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            l_c_c+=1
        '''if  len(t_c) > 0 and len(pd.unique(t_c['track_id'])) > 1 and len(np.where(t_c_check == True)[0]) >=2:
            t_c.to_csv('../interaction_dataset_state/{}/USA_Roundabout_SR_top_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            t_c_c+=1'''
        if  len(r_c) > 0 and len(pd.unique(r_c['track_id'])) > 1 and len(np.where(r_c_check == True)[0]) >=2:
            r_c.to_csv('../interaction_dataset_state/{}/Roundabout_SR_right_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            r_c_c+=1
        if  len(b_c) > 0 and len(pd.unique(b_c['track_id'])) > 1 and len(np.where(b_c_check == True)[0]) >=2:
            b_c.to_csv('../interaction_dataset_state/{}/Roundabout_SR_bottom_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            b_c_c+=1

        # unclump
        # get boolean values where each object has atleast 20 positions
        l_u_check = np.array([len(pd.unique(l_u[l_u['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(l_u['track_id'])])
        #t_u_check = np.array([len(pd.unique(t_u[t_u['track_id'] == id]['frame_id'])) >= 20 for id in
             #        pd.unique(t_u['track_id'])])
        r_u_check = np.array([len(pd.unique(r_u[r_u['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(r_u['track_id'])])
        b_u_check = np.array([len(pd.unique(b_u[b_u['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(b_u['track_id'])])

        if len(l_u) > 0 and len(pd.unique(l_u['track_id'])) > 1 and len(np.where(l_u_check == True)[0]) >=2:
            l_u.to_csv('../interaction_dataset_state/{}/Roundabout_SR_left_unclump_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            l_u_c+=1
        '''if len(t_u) > 0 and len(pd.unique(t_u['track_id'])) > 1 and len(np.where(t_u_check == True)[0]) >=2:
            t_u.to_csv('../interaction_dataset_state/{}/intersection_MA_top_unclump_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            t_u_c+=1'''
        if len(r_u) > 0 and len(pd.unique(r_u['track_id'])) > 1 and len(np.where(r_u_check == True)[0]) >=2:
            r_u.to_csv(
                '../interaction_dataset_state/{}/Roundabout_SR_right_unclump_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            r_u_c+=1
        if len(b_u) > 0 and len(pd.unique(b_u['track_id'])) > 1 and len(np.where(b_u_check == True)[0]) >=2:
            b_u.to_csv(
                '../interaction_dataset_state/{}/Roundabout_SR_bottom_unclump_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            b_u_c+=1

        # neutral
        # get boolean values where each object has atleast 20 positions
        l_n_check = np.array([len(pd.unique(l_n[l_n['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(l_n['track_id'])])
        '''t_n_check = np.array([len(pd.unique(t_n[t_n['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(t_n['track_id'])])'''
        r_n_check = np.array([len(pd.unique(r_n[r_n['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(r_n['track_id'])])
        b_n_check = np.array([len(pd.unique(b_n[b_n['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(b_n['track_id'])])
        if len(l_n) > 0 and len(pd.unique(l_n['track_id'])) > 1 and len(np.where(l_n_check == True)[0]) >=2:
            l_n.to_csv('../interaction_dataset_state/{}/Roundabout_SR_left_neutral_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            l_n_c+=1
        '''if len(t_n) > 0 and len(pd.unique(t_n['track_id'])) > 1 and len(np.where(t_n_check == True)[0]) >=2:
            t_n.to_csv('../interaction_dataset_state/{}/intersection_MA_top_neutral_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            t_n_c+=1'''
        if len(r_n) > 0 and len(pd.unique(r_n['track_id'])) > 1 and len(np.where(r_n_check == True)[0]) >=2:
            r_n.to_csv(
                '../interaction_dataset_state/{}/Roundabout_SR_right_neutral_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            r_n_c+=1
        if len(b_n) > 0 and len(pd.unique(b_n['track_id'])) > 1 and len(np.where(b_n_check == True)[0]) >=2:
            b_n.to_csv(
                '../interaction_dataset_state/{}/Roundabout_SR_bottom_neutral_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            b_n_c+=1

    print("l, t, r, b clump:",l_c_c, t_c_c, r_c_c, b_c_c )
    print("l, t, r, b unclump:", l_u_c, t_u_c, r_u_c, b_u_c)
    print("l, t, r, b neutral:", l_n_c, t_n_c, r_n_c, b_n_c)
    print("clump:",l_c_c+ t_c_c+ r_c_c+ b_c_c )
    print("unclump:",l_u_c+ t_u_c+ r_u_c+ b_u_c )
    print("neutral:",l_n_c+ t_n_c+ r_n_c+ b_n_c )

'''

val data files count:
l, t, r, b clump: 15 0 0 0
l, t, r, b unclump: 17 0 44 6
l, t, r, b neutral: 48 0 20 0
clump: 15
unclump: 67
neutral: 68
'''
