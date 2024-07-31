
import numpy as np


import scipy.sparse as sp
from scipy.sparse import linalg
import torch


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

# I will use the random walk filter
def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, device, pad_with_last_sample=True, shuffle=False):
        """

        :param xs: x values as outputted from seq_io. shape (num_samples, seq_len, num_nodes, num_features)
        :param ys: y values as outputted from seq_io. (num_samples, seq_len, num_nodes, num_features)
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size # how many we need to pad
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs) # number of samples 
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        # xs is a np array of shape (num_samples, seq_len, num_nodes, num_features)
        # each element xs[i] is of shape (seq_len, num_nodes, num_features)
        xs = torch.from_numpy(xs).float()
        ys = torch.from_numpy(ys).float()
        self.xs = xs.to(device)
        self.ys = ys.to(device)

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
import os
import pandas as pd
    
def generate_train_val_test(df, seq_len, horizon, quick_run = True, **data_kwargs):

    # df is the speed data

    validation_ratio = data_kwargs.get('validation_ratio')
    test_ratio = data_kwargs.get('test_ratio')
    num_samples_quick = data_kwargs.get('n_samples_quick')

    if quick_run:
        df = df[:num_samples_quick]


    # x:(epoch_size, seq_len, num_nodes, num_features)
    # y: (epoch_size, horizon, num_nodes, num_features)
#   epoch_size =  num_samples + min(x_offsets) - max(y_offsets)
    x, y = generate_graph_seq2seq_io_data(
        df,
        seq_len= seq_len,
        horizon=horizon, 
        add_time_in_day=True,
        add_day_in_week=False,
    )



    num_samples = x.shape[0] # size of the epoch
    num_test = round(num_samples * test_ratio) 
    num_train = round(num_samples * (1- test_ratio - validation_ratio)) 
    num_val = num_samples - num_test - num_train

    data = {}

    data['x_train']  = x[:num_train] # shape (num_train, seq_len, num_nodes, num_features)
    data['y_train'] = y[:num_train]

    data['x_val'] =   x[num_train: num_train + num_val]
    data['y_val']  = y[num_train: num_train + num_val]
    
    # test
    data['x_test'] = x[-num_test:]
    data['y_test'] = y[-num_test:]

    return data
    


def load_dataset(speed_data, batch_size, device, seq_len, horizon, quick_run, test_batch_size=None,  **data_kwargs):

    # data is loaded which is the output of generate_training_data
    # data is a dictionary containing ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']
    # shapes: x_train (num_train, seq_len, num_nodes, num_features)

    # dropping rows with na – needed when multiple time periods are combined. within one cluster, should always have the same time period
    speed_data = speed_data.dropna()

    data = generate_train_val_test(df = speed_data, seq_len = seq_len, horizon = horizon, quick_run = quick_run,  **data_kwargs)

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # transform the data
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

    
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, device = device, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, device = device, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, device = device, shuffle=False)
    data['scaler'] = scaler

    return data



def generate_graph_seq2seq_io_data(
        df, seq_len, horizon, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df: speed data (clipped to be num_samples_quick length)
    :param seq_len:
    :param horizon: 
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, seq_len, num_nodes, input_dim) where epoch size is the number of data samples per epoch
    # y: (epoch_size, horizon, num_nodes, output_dim)
    """

    
    x_offsets = np.sort(np.concatenate((np.arange(-seq_len + 1, 1, 1),))) # array([-11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0])
    y_offsets = np.sort(np.arange(1, (horizon + 1), 1)) # array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

    

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1) # to shape (num_samples, num_nodes, 1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1) # shape (num_samples, num_nodes, num_features)
    # epoch_size = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets)) 
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...] # x_offsets are negative 
        y_t = data[t + y_offsets, ...] # y_offsets are positive 
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0) # shape (epoch_size, seq_len, num_nodes, num_features)
    y = np.stack(y, axis=0) # shape (epoch_size, horizon, num_nodes, num_features)
    return x, y

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1)) # row sums
    d_inv = np.power(d, -1).flatten() # inverse of row sums
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv) # putting inverse of row sums onto diagonal
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx



