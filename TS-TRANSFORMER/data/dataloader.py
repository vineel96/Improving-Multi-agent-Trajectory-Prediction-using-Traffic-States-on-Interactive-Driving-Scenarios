from data.nuscenes_pred_split import get_nuscenes_pred_split
import os, random, numpy as np, copy
import math

from .preprocessor import preprocess
from .ethucy_split import get_ethucy_split
from utils.utils import print_log


class data_generator(object):

    def __init__(self, parser, log, split='train', phase='training'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred           
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
            self.init_frame = 0
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = parser.data_root_ethucy            
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        elif parser.dataset in {'eot'}:
            ### added ###
            data_root = parser.data_root_eot

            seq_train, seq_test, seq_val = [], [], []
            for file in os.listdir( str(data_root) + '/train/'):
                seq_train.append( str(data_root) + '/train/' + file)

            for file in os.listdir( str(data_root) + '/test/'):
                seq_test.append( str(data_root) + '/test/' + file)

            for file in os.listdir( str(data_root) + '/val/'):
                seq_val.append( str(data_root) + '/val/' + file)

            # sort
            seq_train = sorted(seq_train)
            seq_test = sorted(seq_test)
            seq_val = sorted(seq_val)
            ### added ###
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []

        # self.sequence_to_load: contains list of names of .txt sample files
        # self.num_total_samples: int, total sum of number of sequences (20 length) possible in each .txt sample file
        # self.num_sample_list: list of num of samples present in each .txt sample file
        # self.sequence: list of preprocessor object which contains read data and corresponds to self.num_sample_list indices

        # self.sample_list: list of of numbers, which is used to get the sequence corresponding to a .txt sample file in self.sequence
        ### REFER BELOW EXAMPLE FOR UNDERSTANDING ###
        ### BUG in num_seq_samples calculation ###
        #self.sequence_to_load = ['datasets/EOT_split_preprocessed/train/paldi_clump_5fps_spatial_seq_15_1_1.txt']

        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, parser, log, self.split, self.phase)

            #num_seq_samples = preprocessor.num_fr - (parser.min_past_frames - 1) * self.frame_skip - parser.min_future_frames * self.frame_skip + 1
            num_seq_samples = int( math.ceil((preprocessor.num_fr - 20 + 1) / self.frame_skip)) # similar to SGAN
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
        
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                #frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                # index_tmp is added in above line in order to get the correct sequence, refer example below
                return seq_index, index_tmp #frame_index ( return index_tmp which is base pointer)
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        sample_index = self.sample_list[self.index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        # seq_index corresponds to sample .txt file
        # frame is frame number such that it has 7 past frames and 12 future frames relative to it.
        seq = self.sequence[seq_index]
        self.index += 1
        
        data = seq(frame)  # seq is preprocess class object
        return data      

    def __call__(self):
        return self.next_sample()

    ##### EXAMPLE TO UNDERSTAND DATA GENERATION #####
    '''
    20 frames -> 1, 2, 3,...., 20
    num_seq_samples -> 2 #bug
    
    num_total_samples = 14
    num_sample_list = [2, 3, 4, 5]
    sequence = [o1,o2,o3,o4]
    sample_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    Dry run above code with this example, to understand workflow
    Adding index_tmp to get frame index acts like sliding window, example consider we have 40 frames, seq_len = 20
    num_seq_samples = 40 - 20 = 20
    Now sequences will be: 1,2,...20 || 2,3,....,21 || 3,4,....,22 Here index_tmp acts as displacement value to get each sequence
    '''
