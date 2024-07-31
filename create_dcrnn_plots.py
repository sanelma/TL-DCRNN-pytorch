from lib.plotting import plot_dcrnn_realtime, plot_hourly
import numpy as np
import pandas as pd
import torch
import os
import sys
import matplotlib.pyplot as plt
import pickle
import yaml





# set correct config file here (file that was used for training the outputs)
with open("config_ch.yaml") as f:
    config = yaml.safe_load(f)

input_path = config.get('data_dir')

# set output names here
output_name = "outputs20240704_153644_ZHfull.npy"

speed_data_name = "luzern_counts.h5" # "speed.h5" # "speed_data_subsample.h5"
speed_data_name = "speed_data_subsample.h5"
speed_data_name = "zurich_counts.h5"

# lstm outputs
#lstm_predictions = 'lstm_baseline_predictions_rescale3.npy'
lstm_predictions = ""

days_to_plot = 7




if output_name.endswith(".pkl"):
    with open(f'outputs/{output_name}', 'rb') as f:
        outputs = pickle.load(f)
else:
    outputs = np.load(f'outputs/{output_name}', allow_pickle=True).item()

try: 
    lstm_predictions = np.load(f'outputs/{lstm_predictions}', allow_pickle=True).item()
    predictions = lstm_predictions["prediction"]
    true_values = lstm_predictions['truth']
except:
    print("No LSTM")
    predictions = outputs['prediction']
    true_values = outputs['truth']


   



clusters = list(predictions.keys())
cluster = clusters[0]
cluster = 1

# will only plot for one cluster for now

# y_preds and y_true are lists of length horizon, where each element is size (num_samples, num_nodes)
y_preds = np.array(predictions[cluster])
y_true = np.array(true_values[cluster])





df_test = pd.read_hdf(os.path.join(input_path, speed_data_name))


partition = np.genfromtxt(os.path.join(input_path, config.get('partition_file_name')), dtype=int, delimiter="\n", unpack=False)


sensors = pd.read_csv(os.path.join(input_path, config.get('sensors_file_name' )))



##### Filter sensors

sensors['subgraph'] = partition 
loop_ids = list(df_test.columns)
try:
    loop_ids_num = [int(x) for x in loop_ids]
except: 
    loop_ids_num = loop_ids
sensors = sensors[sensors['sensor_id'].isin(loop_ids_num)]

# then also keep only the first num_nodes sensors from each subgraph
num_nodes = y_preds[0].shape[1]
n_val = y_preds[0].shape[0]
sensors = sensors.groupby('subgraph').head(num_nodes)
partition = sensors['subgraph'].values



sensors = sensors[sensors['subgraph'] == cluster]
sensor_ids = (sensors['sensor_id'].values).astype(str)
sensor_ids = [str(id) for id in sensor_ids]



## read in parameters from config file
model_params = config.get('model')
data_params = config.get('data')
seq_len = model_params.get('seq_len')
n_samples_quick = data_params.get('n_samples_quick')
test_ratio = data_params.get('test_ratio')
horizon = model_params.get('horizon')
test_ratio = data_params.get('test_ratio')
val_ratio = data_params.get('validation_ratio')
num_samples = n_samples_quick - (seq_len - 1) - horizon




# create plots
# steps_into_future is the number of steps into the future that we want to plot
# ex: setting steps_into_future = 0 will plot the models prediction for the upcoming timestamp
# ex: setting steps_into_future = horizon - 1 (due to zero indexing) will plot the models prediction for the most distant future timestamp for which predictions are available
plot_dcrnn_realtime(df_test = df_test, 
          y_true = y_true, 
          y_preds = y_preds, 
          steps_into_future=19, 
          sensor_ids = sensor_ids,
          num_samples = num_samples,
          val_ratio=val_ratio,
          test_ratio=test_ratio,
          seq_len=seq_len,
          horizon=horizon, 
          time_stamps = int(horizon*24*days_to_plot),
          cluster = cluster,
          save_plots = False)

# plot hourly
plot_hourly(df_test = df_test, 
          y_true = y_true, 
          y_preds = y_preds, 
          sensor_ids = sensor_ids,
          num_samples = num_samples,
          val_ratio=val_ratio,
          test_ratio=test_ratio,
          seq_len=seq_len,
          horizon=horizon, 
          hours = int(24*days_to_plot), # set hours to plot the full amount of time
          cluster = cluster,
          save_plots = True)