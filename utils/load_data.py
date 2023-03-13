'''
############################################################################################################################
Discription: 

The two data loader classes are for Korea University dataset and the BCIC-IV-2a dataset. Each loader will
serve Tensor-CSPNet and Graph-CSPNet on two scenairos, i.e., the cross-validation scenario and the holdout scenario. 

Keep in mind that the segmentation plan in this study is a simple example without a deep reason in neurophysiology, 
but achiving a relative good result. More reasonable segmentation plans may yield better performance. 

The class of FilterBank is from https://github.com/ravikiran-mane/FBCNet

#############################################################################################################################
'''

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.signal import cheb2ord
from scipy.linalg import eigvalsh
import torch as th
from pyriemann.estimation import Covariances


class FilterBank:
    def __init__(self, fs, pass_width=4, f_width=4):
        self.fs           = fs
        self.f_trans      = 2
        self.f_pass       = np.arange(4, 40, pass_width)
        self.f_width      = f_width
        self.gpass        = 3
        self.gstop        = 30
        self.filter_coeff = {}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass    = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop    = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp        = f_pass/Nyquist_freq
            ws        = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a      = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff

    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape

        if window_details:
            n_samples = int(self.fs * (window_details.get('tmax') - window_details.get('tmin')))
            #+1

        filtered_data = np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))

        for i, fb in self.filter_coeff.items():
          
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])

            if window_details:
                eeg_data_filtered  = eeg_data_filtered[:,:,int((window_details.get('tmin'))*self.fs):int((window_details.get('tmax'))*self.fs)]
            filtered_data[i,:,:,:] = eeg_data_filtered

        return filtered_data


class load_KU:
    def __init__(self, data_folder, index_folder, alg_name ='Tensor_CSPNet', scenario = 'CV'):

        self.channel_index = [7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        self.alg_name = alg_name
        self.scenario = scenario


        if self.alg_name  == 'Tensor_CSPNet':
            #For Tensor-CSPNet
            self.freq_seg = 4
            self.time_seg =[[0, 1500], [500, 2000], [1000, 2500]]

        elif self.alg_name == 'Graph_CSPNet':
            #For Graph-CSPNet
            self.freq_seg  = 4
            self.time_freq_graph = {
                '1':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '2':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '3':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '4':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '5':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '6':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '7':[[0, 250], [250, 500], [500, 750], [750, 1000],[1000, 1250], [1250,1500],
                     [1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]],
                '8':[[0, 250], [250, 500], [500, 750], [750, 1000],[1000, 1250], [1250,1500],
                     [1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]],
                '9':[[0, 250], [250, 500], [500, 750], [750, 1000],[1000, 1250], [1250,1500],
                     [1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]]
            }
            self.block_dims = [
                          len(self.time_freq_graph['1']), 
                          len(self.time_freq_graph['2']), 
                          len(self.time_freq_graph['3']) + len(self.time_freq_graph['4']) + len(self.time_freq_graph['5']) + len(self.time_freq_graph['6']), 
                          len(self.time_freq_graph['7']) + len(self.time_freq_graph['8']) + len(self.time_freq_graph['9'])
                          ]
            self.time_windows = [5, 5, 5, 10]


        if scenario == 'CV':
            self.x        = np.load(data_folder[0] + '_x.npy')[:, self.channel_index, :]
            self.y_labels = np.load(data_folder[0] + '_y.npy')

            fbank     = FilterBank(fs = 1000, pass_width = self.freq_seg)
            _         = fbank.get_filter_coeff()

            '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
            x_fb = fbank.filter_data(self.x, window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
            self.x_stack = self._tensor_stack(x_fb)
            
            self.train_indices = np.load(index_folder + '_train_index.npy')
            self.test_indices  = np.load(index_folder + '_test_index.npy')

        elif scenario == 'Holdout':

            self.x_1        = np.load(data_folder[0] + '_x.npy')[:, self.channel_index, :]
            self.y_labels_1 = np.load(data_folder[0] + '_y.npy')

            self.x_2        = np.load(data_folder[1] + '_x.npy')[:, self.channel_index, :]
            self.y_labels_2 = np.load(data_folder[1] + '_y.npy')


            fbank    = FilterBank(fs = 1000, pass_width = self.freq_seg)
            _        = fbank.get_filter_coeff()

            x_train_fb = fbank.filter_data(self.x_1,       window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
            x_valid_fb = fbank.filter_data(self.x_2[:100], window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
            x_test_fb  = fbank.filter_data(self.x_2[100:], window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)

            self.x_train_stack = self._tensor_stack(x_train_fb)
            self.x_valid_stack = self._tensor_stack(x_valid_fb)
            self.x_test_stack  = self._tensor_stack(x_test_fb)


    def _tensor_stack(self, x_fb):

        if self.alg_name == 'Tensor_CSPNet':
            '''

            For Tensor-CSPNet:

            Step 1: Segment the signal on each given time intervals.
                    e.g., (trials, frequency bands, channels, timestamps) ---> 
                    (trials, temporal segments, frequency bands, channels, timestamps);
            Step 2: Take covariance.
                    e.g., (trials, temporal segments, frequency bands, channels, timestamps) --->
                    (trials, temporal segments, frequency bands, channels, channels).

            '''
            temporal_seg = []
            for [a, b] in self.time_seg:
                temporal_seg.append(np.expand_dims(x_fb[:, :, :, a:b], axis = 1))
            temporal_seg = np.concatenate(temporal_seg, axis = 1)


            stack_tensor  = []
            for i in range(temporal_seg.shape[0]):
                cov_stack = []
                for j in range(temporal_seg.shape[1]):
                    cov_stack.append(Covariances().transform(temporal_seg[i, j]))
                stack_tensor.append(np.stack(cov_stack, axis = 0))
            stack_tensor = np.stack(stack_tensor, axis = 0)

        elif self.alg_name == 'Graph_CSPNet':
            '''

            For Graph-CSPNet:
            Take covariance on each temporal intervals given in the time-frequency graph. 


            '''
            stack_tensor  = []
            for i in range(1, x_fb.shape[1]+1):
              for [a, b] in self.time_freq_graph[str(i)]:
                cov_record = []
                for j in range(x_fb.shape[0]):
                  cov_record.append(Covariances().transform(x_fb[j, i-1:i, :, a:b]))
                stack_tensor.append(np.expand_dims(np.concatenate(cov_record, axis = 0), axis = 1))
            stack_tensor = np.concatenate(stack_tensor, axis = 1)

        return stack_tensor

    def _riemann_distance(self, A, B):
        #AIRM 
        return np.sqrt((np.log(eigvalsh(A, B))**2).sum())

    def LGT_graph_matrix_fn(self, gamma = 50, time_step = [2, 2, 2, 5], freq_step = [1, 1, 4, 3]):
          '''

          time_step: a list, step of diffusion to right direction.
          freq_step: a list, step of diffusion to down direction.
          gamma: Gaussian coefficent.
          
          '''
          A = np.zeros((sum(self.block_dims), sum(self.block_dims))) + np.eye(sum(self.block_dims))
          start_point = 0
          for m in range(len(self.block_dims)):
            for i in range(self.block_dims[m]):
              max_time_step = min(self.time_windows[m] - 1 - (i % self.time_windows[m]), time_step[m])
              for j in range(i+1, i + max_time_step + 1):
                  A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                  A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
              for freq_mul in range(1, freq_step[m]+1):
                for j in range(i+ freq_mul*self.time_windows[m], i + freq_mul*self.time_windows[m] + max_time_step + 1):
                    if j < self.block_dims[m]: 
                        A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                        A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
            start_point += self.block_dims[m]

          D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis = 0))))

          return np.matmul(D, A), A

    def generate_training_test_set_CV(self, kf_iter):

        train_idx = self.train_indices[kf_iter]
        test_idx  = self.test_indices[kf_iter]

        if self.alg_name == 'Graph_CSPNet':
            self.lattice = np.mean(self.x_stack[train_idx], axis = 0)

        return self.x_stack[train_idx], self.x_stack[test_idx], self.y_labels[train_idx], self.y_labels[test_idx]


    def generate_training_valid_test_set_Holdout(self):

        if self.alg_name == 'Graph_CSPNet':
            self.lattice = np.mean(self.x_train_stack, axis = 0)

        return self.x_train_stack, self.x_valid_stack, self.x_test_stack, self.y_labels_1, self.y_labels_2[:100], self.y_labels_2[100:]


class dataloader_in_main(th.utils.data.Dataset):

    def __init__(self, data_root, data_label):
        self.data  = data_root
        self.label = data_label
 
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
 
    def __len__(self):
        return len(self.data)


def get_data(subject,training,PATH):

    NO_channels   = 22
    NO_tests      = 6*48     
    Window_Length = 7*250 

    class_return = np.zeros(NO_tests)
    data_return  = np.zeros((NO_tests,NO_channels,Window_Length))

    NO_valid_trial = 0
    
    if training:
      a = sio.loadmat(PATH+'A0'+str(subject)+'T.mat')
    else:
      a = sio.loadmat(PATH+'A0'+str(subject)+'E.mat')
    
    a_data = a['data']

    for ii in range(0,a_data.size):
        a_data1     = a_data[0,ii]
        a_data2     = [a_data1[0,0]]
        a_data3     = a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_fs        = a_data3[3]
        a_classes   = a_data3[4]
        a_artifacts = a_data3[5]
        a_gender    = a_data3[6]
        a_age       = a_data3[7]
        for trial in range(0,a_trial.size):
            data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
            class_return[NO_valid_trial]    = int(a_y[trial])
            NO_valid_trial                 += 1

    return data_return, class_return-1 


class load_BCIC:
    def __init__(self, sub, TorE = True, alg_name ='Tensor_CSPNet', session_no = 1, scenario = 'CV'):

        self.alg_name = alg_name
        self.scenario = scenario

        if self.alg_name == 'Tensor_CSPNet':
            self.freq_seg = 4
            self.time_seg =[[0, 625], [375, 1000]]

        elif self.alg_name == 'Graph_CSPNet':
            self.freq_seg  = 4
            self.time_freq_graph = {
            '1':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            '2':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            '3':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            '4':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            '5':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            '6':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            '7':[[0, 125], [125, 250], [250, 375], [375, 500],[500, 625], [625,750],[750, 875], [875, 1000]],
            '8':[[0, 125], [125, 250], [250, 375], [375, 500],[500, 625], [625,750],[750, 875], [875, 1000]],
            '9':[[0, 125], [125, 250], [250, 375], [375, 500],[500, 625], [625,750],[750, 875], [875, 1000]]
            }
            self.block_dims = [
                          len(self.time_freq_graph['1']), 
                          len(self.time_freq_graph['2']), 
                          len(self.time_freq_graph['3']) + len(self.time_freq_graph['4']) + len(self.time_freq_graph['5']) + len(self.time_freq_graph['6']), 
                          len(self.time_freq_graph['7']) + len(self.time_freq_graph['8']) + len(self.time_freq_graph['9'])
                          ]
            self.time_windows = [4, 4, 4, 8]

        if scenario == 'CV':
            self.x, self.y_labels = get_data(sub, TorE, 'dataset/')
            fbank     = FilterBank(fs = 250, pass_width = self.freq_seg)
            _         = fbank.get_filter_coeff()

            '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
            x_fb         = fbank.filter_data(self.x, window_details={'tmin':3.0, 'tmax':7.0}).transpose(1, 0, 2, 3)
            self.x_stack = self._tensor_stack(x_fb)
            
            self.train_indices = np.load('index/BCIC_index/sess'+str(session_no)+'_sub'+str(sub)+'_train_index.npy', allow_pickle=True)
            self.test_indices  = np.load('index/BCIC_index/sess'+str(session_no)+'_sub'+str(sub)+'_test_index.npy', allow_pickle=True)

        elif scenario == 'Holdout':

            self.x_1, self.y_labels_1 = get_data(sub, True, 'dataset/')
            self.x_2, self.y_labels_2 = get_data(sub, False, 'dataset/')

            fbank    = FilterBank(fs = 250, pass_width = self.freq_seg)
            _        = fbank.get_filter_coeff()

            x_train_fb = fbank.filter_data(self.x_1, window_details={'tmin':3.0, 'tmax':7.0}).transpose(1, 0, 2, 3)
            x_test_fb  = fbank.filter_data(self.x_2, window_details={'tmin':3.0, 'tmax':7.0}).transpose(1, 0, 2, 3)

            self.x_train_stack = self._tensor_stack(x_train_fb)
            self.x_test_stack  = self._tensor_stack(x_test_fb)



    def _tensor_stack(self, x_fb):

        if self.alg_name == 'Tensor_CSPNet':
            '''
            For Tensor-CSPNet:

            Step 1: Segment the signal on each given time intervals.
                    e.g., (trials, frequency bands, channels, timestamps) ---> 
                    (trials, temporal segments, frequency bands, channels, timestamps);
            Step 2: Take covariance.
                    e.g., (trials, temporal segments, frequency bands, channels, timestamps) --->
                    (trials, temporal segments, frequency bands, channels, channels).
            '''
            temporal_seg   = []
            for [a, b] in self.time_seg:
                temporal_seg.append(np.expand_dims(x_fb[:, :, :, a:b], axis = 1))
            temporal_seg   = np.concatenate(temporal_seg, axis = 1)

            stack_tensor   = []
            for i in range(temporal_seg.shape[0]):
                cov_stack  = []
                for j in range(temporal_seg.shape[1]):
                    cov_stack.append(Covariances().transform(temporal_seg[i, j]))
                stack_tensor.append(np.stack(cov_stack, axis = 0))
            stack_tensor   = np.stack(stack_tensor, axis = 0)

        elif self.alg_name == 'Graph_CSPNet':
            '''
            For Graph-CSPNet:
            Take covariance on each temporal intervals given in the time-frequency graph. 

            '''
            stack_tensor   = []
            for i in range(1, x_fb.shape[1]+1):
              for [a, b] in self.time_freq_graph[str(i)]:
                cov_record = []
                for j in range(x_fb.shape[0]):
                  cov_record.append(Covariances().transform(x_fb[j, i-1:i, :, a:b]))
                stack_tensor.append(np.expand_dims(np.concatenate(cov_record, axis = 0), axis = 1))
            stack_tensor   = np.concatenate(stack_tensor, axis = 1)

        return stack_tensor

    def _riemann_distance(self, A, B):
        #geodesic distance under metric AIRM. 
        return np.sqrt((np.log(eigvalsh(A, B))**2).sum())

    def LGT_graph_matrix_fn(self, gamma = 50, time_step = [2, 2, 2, 4], freq_step = [1, 1, 4, 3]):
          '''
          time_step: a list, step of diffusion to right direction.
          freq_step: a list, step of diffusion to down direction.
          gamma: Gaussian coefficent.
          '''
        
          A = np.zeros((sum(self.block_dims), sum(self.block_dims))) + np.eye(sum(self.block_dims))
          start_point = 0
          for m in range(len(self.block_dims)):
            for i in range(self.block_dims[m]):
              max_time_step = min(self.time_windows[m] - 1 - (i % self.time_windows[m]), time_step[m])
              for j in range(i + 1, i + max_time_step + 1):
                  A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                  A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
              for freq_mul in range(1, freq_step[m]+1):
                for j in range(i+ freq_mul*self.time_windows[m], i + freq_mul*self.time_windows[m] + max_time_step + 1):
                    if j < self.block_dims[m]: 
                        A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                        A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
            start_point += self.block_dims[m]

          D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis = 0))))

          return np.matmul(D, A), A


    def generate_training_test_set_CV(self, kf_iter):

        train_idx = self.train_indices[kf_iter]
        test_idx  = self.test_indices[kf_iter]

        if self.alg_name == 'Graph_CSPNet':
            self.lattice = np.mean(self.x_stack[train_idx], axis = 0)

        return self.x_stack[train_idx], self.x_stack[test_idx], self.y_labels[train_idx], self.y_labels[test_idx]


    def generate_training_valid_test_set_Holdout(self):

        if self.alg_name == 'Graph_CSPNet':
            self.lattice = np.mean(self.x_train_stack, axis = 0)

        return self.x_train_stack, self.x_test_stack, self.y_labels_1, self.y_labels_2



