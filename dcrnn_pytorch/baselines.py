import argparse
import numpy as np
import pandas as pd
import torch
from torch.nn import LSTM
import sys

from statsmodels.tsa.vector_ar.var_model import VAR

from dcrnn_pytorch.dcrnn_utils import StandardScaler, generate_graph_seq2seq_io_data, DataLoader




# baseline as a average over the previous horizon time stamps
def moving_avg_baseline(xs, horizon): 
    
    """Computes averages over previous hour as predictions, in order to compare as a baseline
    NOTE : The number of samples will be off by one, since we do not have the preceding hour. But still, probably decent baseline for now
    Inputs:
    - xs tensor of the shape (batch_size, seq_len, num_sensor, input_dim) (as xs from data loader)
    - Return tensor shape (horizon, batch_size, num_nodes * output_dim), like model output
    """

    # xs is (batch_size, seq_len, num_sensor, input_dim) (b, 12, 10, 1)
    batch_size = xs.shape[0]
    seq_len = xs.shape[1]
    num_sensor = xs.shape[2]
    input_dim = xs.shape[3]
    output_dim = 1

    # compute average for each sensor over the sequence length: (batch_size, num_sensor, input_dim
    means = torch.mean(xs, dim=1)
    means_reshaped = means.view(batch_size, num_sensor * input_dim)
    # Repeat the averages horizon many times and reshape to the desired shape
    repeated_means = means_reshaped.unsqueeze(0).repeat(horizon, 1, 1)

    # want to return (horizon, batch_size, num_nodes * output_dim)
    repeated_means = repeated_means.permute(0, 1, 2)

    return repeated_means


# just take the most recent timestamp
def impute_previous_baseline(xs, horizon): 
    
    """Computes averages over previous hour as predictions, in order to compare as a baseline
    NOTE : The number of samples will be off by one, since we do not have the preceding hour. But still, probably decent baseline for now
    Inputs:
    - xs tensor of the shape (batch_size, seq_len, num_sensor, input_dim) (as xs from data loader)
    - Return tensor shape (horizon, batch_size, num_nodes * output_dim), like model output
    """

    # xs is (batch_size, seq_len, num_sensor, input_dim) (b, 12, 10, 1)
    batch_size = xs.shape[0]
    seq_len = xs.shape[1]
    num_sensor = xs.shape[2]
    input_dim = xs.shape[3]
    output_dim = 1

    # compute average for each sensor over the sequence length: (batch_size, num_sensor, input_dim
    previous = xs[:, -1, :, :]
    previous_reshaped = previous.view(batch_size, num_sensor * input_dim)
    # Repeat the averages horizon many times and reshape to the desired shape
    forecast = previous_reshaped.unsqueeze(0).repeat(horizon, 1, 1)

    # want to return (horizon, batch_size, num_nodes * output_dim)
    forecast = forecast.permute(0, 1, 2)

    return forecast



def var_baseline(df, n_samples_quick, sensor_ids,seq_len, horizon, val_ratio = 0.1, test_ratio = 0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    : Inputs: 
    - ## Nope: xs tensor of the shape (batch_size, seq_len, num_sensor, input_dim) (as xs from data loader)
    - df: speed dataframe
    - n_samples_quick: number of samples set in config file 
    - sensor_ids: list of sensor ids
    - val_ratio, test_ratio, seq_len, horizon
    :return: 
    - df_predicts: list of length horizon, each element pandas df of size (n_test, num_sensor)
    - df_test: true test dataframe of size (n_test, num_sensor)
    """
    df = df.loc[:, sensor_ids]
    n_samples_quick = min(len(df), n_samples_quick)

    num_sensor = len(sensor_ids) # problem maybe when i only use part of the data
 
    output_dim = 1

    num_samples = n_samples_quick - (seq_len - 1) - horizon

    train_ratio = 1 - val_ratio - test_ratio

    n_val = round(num_samples*val_ratio)
    n_train = round(num_samples*train_ratio) 
    n_test = round(num_samples * test_ratio) 

    # start_index in speed data. 

    start_index = int(n_train) + int(n_val) 
    end_index = int(n_train) + int(n_val) + n_test 

    df_train = df.loc[:, sensor_ids].iloc[:n_train, :]
    # starts at the index where we start forecasting.
    # since the forecasts are for the future, the true values that correspond will be shifted up by seq_len many time stamps
    df_test = df.loc[:, sensor_ids].iloc[start_index:end_index, :] 



    # fit model 
    scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
    data = scaler.transform(df_train.values)
    var_model = VAR(data)
    var_result = var_model.fit(seq_len)


    # Do forecasting.
    result = np.zeros(shape=(horizon, n_test, num_sensor))
    start = n_train - seq_len - horizon + 1 #same as num_samples
    for input_ind in range(len(df_test) - seq_len):
        prediction = var_result.forecast(scaler.transform(df_test.values[input_ind: input_ind + seq_len]), horizon)   # prediction shape is (horizon, num_sensor) 
        for n_forward in range(horizon):
            result[n_forward, input_ind, :] = prediction[n_forward, :] # size (num_nodes, 1). gets inserted into result at the given index and timestamp of horizon

    df_predicts = []
    for n_forward in range(horizon):
        df_predict = pd.DataFrame(scaler.inverse_transform(result[n_forward]), index=df_test.index, columns=df_test.columns)
        df_predicts.append(df_predict)
        
    # get the y values that correspond to the same time stamps as the prediction
    df_true = df.loc[:, sensor_ids].iloc[start_index+seq_len:end_index+seq_len, :]  
    return df_predicts, df_true



# compute the baselines for a given cluster (given my the sensor ids)
def generate_baseline_predictions(df_test, sensor_ids, n_samples_quick, test_ratio, val_ratio,
                                  seq_len, horizon):
    
    df_test = df_test.loc[:, sensor_ids]
    n_samples_quick = min(len(df_test), n_samples_quick)
    # generate xs and ys from data loader for cluster
    xs, ys = generate_graph_seq2seq_io_data(df_test.loc[:, sensor_ids][:n_samples_quick], seq_len = seq_len,
                                         horizon = horizon, add_time_in_day=False)
    
    num_samples = n_samples_quick - (seq_len - 1) - horizon
    num_test = round(num_samples * test_ratio)
    num_train = round(num_samples * (1 - test_ratio - val_ratio))

    # xs is (batch_size, seq_len, num_sensor, input_dim)
    xs_test = xs[-num_test:]
    ys_test = ys[-num_test:]

    xs_train = xs[num_train:]
    ys_train = ys[num_train:]

    # create data loader
    test_loader = DataLoader(xs_test,ys_test, batch_size=1, device = "cpu", shuffle=False)
    test_iterator = test_loader.get_iterator()


    y_true_avg = []
    y_preds_avg = []
    y_preds_previous = []



    for _, (x, y) in enumerate(test_iterator):
    # shape of x and y is [1, 12, 10, 1] (batch_size, seq_len, num_sensor, input_dim)
    # change y from (batch_size, horizon, num_sensor, output_dim) to (seq_len, batch_size, num_sensor, output_dim)

        y = y.permute(1, 0, 2, 3)

        # next change y from (seq_len, batch_size, num_sensor, output_dim) to (seq_len, batch_size, num_sensor * output_dim)
        batch_size = y.size(1)
        horizon = y.size(0)
        num_nodes = y.size(2)
        output_dim = y.size(3)
        
        y = y[..., :output_dim].view(horizon, batch_size,
                                            num_nodes * output_dim)

        output = moving_avg_baseline(x, horizon) # output shape (horizon, batch_size, num_nodes * output_dim)
        output_previous = impute_previous_baseline(x, horizon)
    


        y_true_avg.append(y) # y_preds is a list of length batches_per_epoch, where each element is a tensor of shape (horizon, batch_size, num_nodes * output_dim)
        y_preds_avg.append(output)
        y_preds_previous.append(output_previous)
        

             
    y_preds_avg = np.concatenate(y_preds_avg, axis=1) # (horizon, epoch_size, num_nodes)
    y_preds_previous = np.concatenate(y_preds_previous, axis=1) 
    y_true_avg = np.concatenate(y_true_avg, axis=1)  # concatenate on batch dimension

    return y_preds_avg, y_preds_previous, y_true_avg


def check_if_true_values_align(df_true, y_true):
    for c in range(len(df_true.columns)):
        y_true_current_list = [round(float(num), 2) for num in df_true.iloc[:, c].tolist()]
        y_true_output_list = [round(float(num), 2) for num in y_true[0, :, c]]

        if  y_true_current_list != y_true_output_list:
            print('True values do not align')
            print(df_true)
            print(y_true)
            sys.exit("Error with indexing or cluster ids")  



def get_sensors_with_largest_error(y_true, y_preds, loss_function):
    """
    Inputs:
    - y_true for one cluster as a numpy array of size (horizon, num_samples, num_nodes)
    - y_preds for one cluster as a numpy array of size (horizon, num_samples, num_nodes)
    - loss function
    Output:
    - errors: numpy array of size (num_nodes) with the average error for each node
    """
    num_nodes = y_true.shape[2]
    errors = np.zeros(num_nodes)

    for node in range(num_nodes):
        errors[node] = loss_function(y_true[:, :, node], y_preds[:, :, node])

    return errors # return indices of nodes with largest errors




         


