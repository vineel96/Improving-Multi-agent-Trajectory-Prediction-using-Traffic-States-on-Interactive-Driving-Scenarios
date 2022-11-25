import pandas as pd
import numpy as np
#pd.set_option("display.max_rows", None, "display.max_columns", None)

'''we treat val data as test data for evaluation purposes'''

if __name__=="__main__":
    full_data_path = '../recorded_trackfiles/train/DR_USA_Intersection_GL_train.csv'
    data = pd.read_csv(full_data_path, sep=',', index_col=False, header=0)  # change delimiter seperator
    data.sort_values('case_id', inplace=True)
    count = 0
    mode = "train"
    l_c_c, t_c_c, r_c_c, b_c_c =0, 0, 0, 0
    l_u_c, t_u_c, r_u_c, b_u_c =0, 0, 0, 0
    l_n_c, t_n_c, r_n_c, b_n_c =0, 0, 0, 0
    for case_id in pd.unique(data['case_id']):
        print("processing case id: ",case_id)
        l_c, t_c,  b_c = [], [], []  # clump region at different junctions
        l_u, t_u,  b_u = [], [], [] # unclump region at different junctions
        l_n, t_n,  b_n = [], [], []  # neutral region at different junctions
        d = data[ data['case_id'] == case_id ]

        d = d[ d['agent_type'] == 'car']
        d.sort_values('frame_id', inplace=True)
        d = d[ ["frame_id", "track_id", "x", "y"] ] # filter only required columns
        if len(d) == 0:
            continue

        #clump check
        l_c = d[ (d['x'] >= 942) & (d['x'] <= 942 + 14) & (d['y'] >= 993) & (d['y'] <= 993 + 8)] # left junction
        t_c = d[ (d['x'] >= 1021) & (d['x'] <= 1021 + 10 ) & (d['y'] >= 1017) & (d['y'] <= 1017 + 14) ] # top   junction
        #r_c = d[ (d['x'] >= 1017) & (d['x'] <= 1017 + 19 ) & (d['y'] >= 1006) & (d['y'] <= 1006 + 5) ] # right   junction
        b_c = d[ (d['x'] >= 987) & (d['x'] <= 987 + 10 ) & (d['y'] >= 953) & (d['y'] <= 953 + 10) ] # bottom   junction
        # unclump check
        l_u = d[ (d['x'] >= 975) & (d['x'] <= 975 + 17) & (d['y'] >= 985) & (d['y'] <= 985 + 10)] # left junction
        t_u = d[ (d['x'] >= 1011) & (d['x'] <= 1011 + 16) & (d['y'] >= 989) & (d['y'] <= 989 + 13) ] # top   junction
        #r_u = d[ (d['x'] >= 1024) & (d['x'] <= 1024 + 15) & (d['y'] >= 1006) & (d['y'] <= 1006 + 6)] # right   junction
        b_u = d[ (d['x'] >= 977) & (d['x'] <= 977 + 20) & (d['y'] >= 984) & (d['y'] <= 984 + 13) ] # bottom   junction
        # neutral check
        l_n = d[ (d['x'] >= 956) & (d['x'] <= 956 + 18) & (d['y'] >= 991) & (d['y'] <= 991 + 9) ] # left junction
        t_n = d[ (d['x'] >= 1017) & (d['x'] <= 1017 + 13) & (d['y'] >= 1005) & (d['y'] <= 1005 + 9) ] # top   junction
        #r_n = d[ (d['x'] >= 1038) & (d['x'] <= 1038 + 14) & (d['y'] >= 1006) & (d['y'] <= 1006 + 5) ] # right   junction
        b_n = d[ (d['x'] >= 984) & (d['x'] <= 984 + 14) & (d['y'] >= 965) & (d['y'] <= 965 + 13) ] # bottom   junction

        # clump
        # get boolean values where each object has atleast 20 positions
        l_c_check = np.array([ len(pd.unique(l_c[ l_c['track_id'] == id ]['frame_id']))>=20 for id in pd.unique(l_c['track_id']) ])
        t_c_check = np.array([len(pd.unique(t_c[t_c['track_id'] == id]['frame_id'])) >= 20 for id in pd.unique(t_c['track_id'])])
        #r_c_check = np.array([len(pd.unique(r_c[r_c['track_id'] == id]['frame_id'])) >= 20 for id in pd.unique(r_c['track_id'])])
        b_c_check = np.array([len(pd.unique(b_c[b_c['track_id'] == id]['frame_id'])) >= 20 for id in pd.unique(b_c['track_id'])])

        if  len(l_c) > 0 and len(pd.unique(l_c['track_id'])) > 1 and len(np.where(l_c_check == True)[0]) >=2:
            l_c.to_csv('../interaction_dataset_state/{}/intersection_GL_left_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            l_c_c+=1
        if  len(t_c) > 0 and len(pd.unique(t_c['track_id'])) > 1 and len(np.where(t_c_check == True)[0]) >=2:
            t_c.to_csv('../interaction_dataset_state/{}/intersection_GL_top_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            t_c_c+=1
        '''if  len(r_c) > 0 and len(pd.unique(r_c['track_id'])) > 1 and len(np.where(r_c_check == True)[0]) >=2:
            r_c.to_csv('../interaction_dataset_state/{}/intersection_GL_right_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            r_c_c+=1'''
        if  len(b_c) > 0 and len(pd.unique(b_c['track_id'])) > 1 and len(np.where(b_c_check == True)[0]) >=2:
            b_c.to_csv('../interaction_dataset_state/{}/intersection_GL_bottom_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            b_c_c+=1

        # unclump
        # get boolean values where each object has atleast 20 positions
        l_u_check = np.array([len(pd.unique(l_u[l_u['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(l_u['track_id'])])
        t_u_check = np.array([len(pd.unique(t_u[t_u['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(t_u['track_id'])])
        '''r_u_check = np.array([len(pd.unique(r_u[r_u['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(r_u['track_id'])])'''
        b_u_check = np.array([len(pd.unique(b_u[b_u['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(b_u['track_id'])])

        if len(l_u) > 0 and len(pd.unique(l_u['track_id'])) > 1 and len(np.where(l_u_check == True)[0]) >=2:
            l_u.to_csv('../interaction_dataset_state/{}/intersection_GL_left_unclump_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            l_u_c+=1
        if len(t_u) > 0 and len(pd.unique(t_u['track_id'])) > 1 and len(np.where(t_u_check == True)[0]) >=2:
            t_u.to_csv('../interaction_dataset_state/{}/intersection_GL_top_unclump_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            t_u_c+=1
        '''if len(r_u) > 0 and len(pd.unique(r_u['track_id'])) > 1 and len(np.where(r_u_check == True)[0]) >=2:
            r_u.to_csv(
                '../interaction_dataset_state/{}/intersection_MA_right_unclump_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',') 
            r_u_c+=1'''
        if len(b_u) > 0 and len(pd.unique(b_u['track_id'])) > 1 and len(np.where(b_u_check == True)[0]) >=2:
            b_u.to_csv(
                '../interaction_dataset_state/{}/intersection_GL_bottom_unclump_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            b_u_c+=1

        # neutral
        # get boolean values where each object has atleast 20 positions
        l_n_check = np.array([len(pd.unique(l_n[l_n['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(l_n['track_id'])])
        t_n_check = np.array([len(pd.unique(t_n[t_n['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(t_n['track_id'])])
        '''r_n_check = np.array([len(pd.unique(r_n[r_n['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(r_n['track_id'])])'''
        b_n_check = np.array([len(pd.unique(b_n[b_n['track_id'] == id]['frame_id'])) >= 20 for id in
                     pd.unique(b_n['track_id'])])
        if len(l_n) > 0 and len(pd.unique(l_n['track_id'])) > 1 and len(np.where(l_n_check == True)[0]) >=2:
            l_n.to_csv('../interaction_dataset_state/{}/intersection_GL_left_neutral_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            l_n_c+=1
        if len(t_n) > 0 and len(pd.unique(t_n['track_id'])) > 1 and len(np.where(t_n_check == True)[0]) >=2:
            t_n.to_csv('../interaction_dataset_state/{}/intersection_GL_top_neutral_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            t_n_c+=1
        '''if len(r_n) > 0 and len(pd.unique(r_n['track_id'])) > 1 and len(np.where(r_n_check == True)[0]) >=2:
            r_n.to_csv(
                '../interaction_dataset_state/{}/intersection_MA_right_neutral_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            r_n_c+=1'''
        if len(b_n) > 0 and len(pd.unique(b_n['track_id'])) > 1 and len(np.where(b_n_check == True)[0]) >=2:
            b_n.to_csv(
                '../interaction_dataset_state/{}/intersection_GL_bottom_neutral_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            b_n_c+=1

    print("l, t, r, b clump:",l_c_c, t_c_c, r_c_c, b_c_c )
    print("l, t, r, b unclump:", l_u_c, t_u_c, r_u_c, b_u_c)
    print("l, t, r, b neutral:", l_n_c, t_n_c, r_n_c, b_n_c)
    print("clump:",l_c_c+ t_c_c+ r_c_c+ b_c_c )
    print("unclump:",l_u_c+ t_u_c+ r_u_c+ b_u_c )
    print("neutral:",l_n_c+ t_n_c+ r_n_c+ b_n_c )

'''
train data files count:
l, t, r, b clump: 205 99 521 51
l, t, r, b unclump: 1024 304 1 814
l, t, r, b neutral: 3224 1709 647 962
clump: 876
unclump: 2143
neutral: 6542

val data files count:
l, t, r, b clump: 6 9 1 1
l, t, r, b unclump: 189 57 0 170
l, t, r, b neutral: 638 382 27 229
clump: 17
unclump: 416
neutral: 1276
'''
