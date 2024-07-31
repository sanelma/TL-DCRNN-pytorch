
import numpy as np
import pandas as pd
import torch
import os
import sys
import matplotlib.pyplot as plt
import pickle
import logging
import yaml

from lib.metrics import masked_mse_np, masked_mape_np, masked_mae_np, masked_rmse_np, masked_mape_np

from dcrnn_pytorch.baselines import generate_baseline_predictions, var_baseline, check_if_true_values_align
from lib.plotting import plot_dcrnn_realtime, plot_hourly





# set output names here

output_name = "outputs20240704_153644_ZHfull.npy"
output_name_dcrnn = "outputs20240709_113335_trainOnLuzern.npy" # baseline dcrnn model trained on the target dataset

# whether we also want to calculate the VAR baseline - slower training time
calculate_var = False

# set correct config file here
with open("config_ch.yaml") as f:
    config = yaml.safe_load(f)




if output_name.endswith(".pkl"):
    with open(f'outputs/{output_name}', 'rb') as f:
        outputs = pickle.load(f)
else:
    outputs = np.load(f'outputs/{output_name}', allow_pickle=True).item()

if output_name_dcrnn.endswith(".pkl"):
    with open(f'outputs/{output_name_dcrnn}', 'rb') as f:
        outputs_dcrnn = pickle.load(f)
else:
    outputs_dcrnn = np.load(f'outputs/{output_name_dcrnn}', allow_pickle=True).item()

print("Loaded outputs")

# configure logging
logging.basicConfig(filename=f"logs/evaluate_baselines_{output_name}.log", level=logging.INFO, filemode='w', 
                        format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging is configured.")

logging.info(f"Evaluating outputs {output_name}")

## read in parameters from config file
input_path = config.get('data_dir')
model_params = config.get('model')
data_params = config.get('data')
seq_len = model_params.get('seq_len')
n_samples_quick = data_params.get('n_samples_quick')
test_ratio = data_params.get('test_ratio')
horizon = model_params.get('horizon')
test_ratio = data_params.get('test_ratio')
val_ratio = data_params.get('validation_ratio')
num_samples = n_samples_quick - (seq_len - 1) - horizon


    

predictions = outputs['prediction']
true_values = outputs['truth']

predictions_dcrnn_ntl = outputs_dcrnn['prediction']
true_values_dcrnn_ntl = outputs_dcrnn['truth']

for cluster in predictions.keys():
    assert np.allclose(true_values[cluster], true_values_dcrnn_ntl[cluster]), "True values do not align between model trained with and without transfer learning"

test_clusters = list(predictions.keys())


df_test = pd.read_hdf(os.path.join(input_path, config.get('traffic_file_name')))
df_test = df_test[:n_samples_quick]

try:
    speed_data_target = pd.read_hdf(os.path.join(input_path, config.get('target_traffic_file_name')))
    df_test = pd.concat([df_test, speed_data_target], axis=1) 
    df_test.index = pd.to_datetime(df_test.index)
except:
    pass


partition = np.genfromtxt(os.path.join(input_path, config.get('partition_file_name')), dtype=int, delimiter="\n", unpack=False)
sensors_all = pd.read_csv(os.path.join(input_path, config.get('sensors_file_name' )))


#### In case the train and test data are in different files, read in the test data and combine it





##### Filter sensors
sensors_all['subgraph'] = partition 
loop_ids = list(df_test.columns)
try:
    loop_ids_num = [int(x) for x in loop_ids]
except: 
    loop_ids_num = loop_ids
sensors_all = sensors_all[sensors_all['sensor_id'].isin(loop_ids_num)] 





# just taking y_preds and y_true for the first cluster here
y_preds = predictions[test_clusters[0]]
y_true = true_values[test_clusters[0]]

# size (horizon, num_samples, num_nodes)
y_true = np.array(y_true)
y_preds = np.array(y_preds)

# keep only the first num_nodes sensors from each subgraph
# this should be the same for each subgraph
num_nodes = y_preds[0].shape[1]
n_val = y_preds[0].shape[0]
sensors_all = sensors_all.groupby('subgraph').head(num_nodes)
partition = sensors_all['subgraph'].values


sensor_ids_all = (sensors_all['sensor_id'].values).astype(str)
sensor_ids_all = [str(id) for id in sensor_ids_all]




print("Processed data")
logging.info("Processed_data")



predictions_avg = {}
predictions_previous= {}
predictions_var = {}
predictions_model = {}
predictions_dcrnn = {}
truths = {}


for cluster_ind in range(len(test_clusters)):
    
    cluster_id = test_clusters[cluster_ind]
    print("Generating predictions for cluster", cluster_id)
    logging.info(f"Generating predictions for cluster {cluster_id}")

    # sensor ids for the cluster
    sensors_cluster = sensors_all[sensors_all['subgraph'] == cluster_id]
    sensor_ids_cluster = (sensors_cluster['sensor_id'].values).astype(str)
    sensor_ids_cluster = [str(id) for id in sensor_ids_cluster]

    df_test_sensors = df_test[sensor_ids_cluster]
    df_test_sensors = df_test_sensors.dropna()

    # generate the baseline predictions
    y_preds_avg, y_preds_previous, y_true_avg = generate_baseline_predictions(df_test_sensors, sensor_ids = sensor_ids_cluster, 
                                                                                n_samples_quick = n_samples_quick, 
                                                                                test_ratio = test_ratio, val_ratio = val_ratio,
                                                                                seq_len = seq_len, horizon= horizon)
    
    logging.info("Generated baseline predictions for average and previous")
                                                                                
    if calculate_var:
        df_test_sensors_cleaned = df_test_sensors.loc[:, (df_test_sensors != 0).any(axis=0)]
        df_predicts_var, df_true_var = var_baseline(df_test_sensors_cleaned, n_samples_quick = n_samples_quick,
                                                    sensor_ids = list(df_test_sensors_cleaned.columns), 
                                                    seq_len = seq_len, horizon = horizon,
                                                    val_ratio = val_ratio, test_ratio = test_ratio)
        logging.info("Generated baseline predictions for VAR")
     #   check_if_true_values_align(df_true_var, y_true_avg)
      #  check_if_true_values_align(df_true_var, np.array(true_values[cluster_id]))
    # if don't want to calculate var, just set to be zeros 
    else:
        df_predicts_var = np.zeros_like(y_preds_avg)
        df_true_var = y_true_avg
        logging.info("Not calculating var, set to zeros")

    
    
    


    # save into the dictionary that stores the predictions
    # keys are cluster ids, and values are np arrays of size (horizon, num_samples, num_nodes)

    predictions_model_cluster = np.array(predictions[cluster_id])
    predictions_dcrnn_ntl_cluster = np.array(predictions_dcrnn_ntl[cluster_id])

    # set negative values to 0
    predictions_model_cluster[predictions_model_cluster < 0] = 0
    predictions_dcrnn_ntl_cluster[predictions_dcrnn_ntl_cluster < 0] = 0
    

    predictions_avg[cluster_id] = y_preds_avg
    predictions_previous[cluster_id] = y_preds_previous 
    predictions_var[cluster_id] = np.array(df_predicts_var)
    predictions_model[cluster_id] = predictions_model_cluster
    predictions_dcrnn[cluster_id] = predictions_dcrnn_ntl_cluster
    truths[cluster_id] = np.array(true_values[cluster_id])


all_predictions = {"Model": predictions_model, 
                   "Average": predictions_avg, 
                   "DCRNN": predictions_dcrnn,
                   "Previous": predictions_previous,
                     "VAR": predictions_var, 
                     "True": truths}

with open(f'outputs/baseline_predictions_{output_name}.pkl', 'wb') as f:
    pickle.dump(all_predictions, f)


print("Computed all baseline predictions")
logging.info("Computed all baseline predictions")

#sys.exit("Finished computing baseline predictions")

# computing the losses. there are per cluster and time stamp. function where you pass in the loss function
def compute_losses(loss_fun):
    print("Computing losses", loss_fun.__name__)
    logging.info(f"Computing losses {loss_fun.__name__}")

    # initialize arrays of losses, of size (horizon, num_clusters), one for each baseline
    losses_avg = np.zeros((horizon, len(test_clusters)))
    losses_previous = np.zeros((horizon, len(test_clusters)))
    losses_var = np.zeros((horizon, len(test_clusters)))
    losses_model = np.zeros((horizon, len(test_clusters)))
    losses_dcrnn = np.zeros((horizon, len(test_clusters)))

    for time_stamp in range(horizon):
        for cluster_ind in range(len(test_clusters)):

            # retrieve the predictions
            cluster_id = test_clusters[cluster_ind]
            y_preds_previous = predictions_previous[cluster_id]
            y_preds_avg = predictions_avg[cluster_id]
            y_preds_model = predictions_model[cluster_id]
            y_preds_var = predictions_var[cluster_id]
            y_trues = truths[cluster_id]

            # compute the losses            
            loss_model = loss_fun(y_trues[time_stamp], y_preds_model[time_stamp], 0)
            loss_avg = loss_fun(y_trues[time_stamp], y_preds_avg[time_stamp], 0)
            loss_previous = loss_fun(y_trues[time_stamp], y_preds_previous[time_stamp], 0)
            loss_dcrnn = loss_fun(y_trues[time_stamp], predictions_dcrnn[cluster_id][time_stamp], 0)
            loss_var = loss_fun(y_trues[time_stamp], y_preds_var[time_stamp], 0)

            # put the losses into the loss arrays of size (horizon, num_clusters)
            # the arrays don't have the cluster ids, but they are in the same order as the test_clusters
            losses_avg[time_stamp, cluster_ind] = loss_avg
            losses_previous[time_stamp, cluster_ind] = loss_previous
            losses_model[time_stamp, cluster_ind] = loss_model
            losses_dcrnn[time_stamp, cluster_ind] = loss_dcrnn
            losses_var[time_stamp, cluster_ind] = loss_var


    # take average over columns (clusters) to get average loss per time stamp
    # the losses arrays are of size (horizon, num_clusters)
    avg_losses_avg = np.mean(losses_avg, axis = 1).squeeze()
    avg_losses_previous = np.mean(losses_previous, axis = 1).squeeze()
    avg_losses_model = np.mean(losses_model, axis = 1).squeeze()
    avg_losses_dcrnn = np.mean(losses_dcrnn, axis = 1).squeeze()
    avg_losses_var = np.mean(losses_var, axis = 1).squeeze()

    losses_by_timestamp = {"Clusters": test_clusters,
                "Loss function": loss_fun,
                "DCRNN-TL": avg_losses_model, 
                "DCRNN": avg_losses_dcrnn,
                "Average": avg_losses_avg, 
                "Previous": avg_losses_previous, 
                "VAR": avg_losses_var}
    
    # take average over rows (clusters) to get average loss per cluster
    avg_losses_avg = np.mean(losses_avg, axis = 0).squeeze()
    avg_losses_previous = np.mean(losses_previous, axis = 0).squeeze()
    avg_losses_model = np.mean(losses_model, axis = 0).squeeze()
    avg_losses_dcrnn = np.mean(losses_dcrnn, axis = 0).squeeze()
    avg_losses_var = np.mean(losses_var, axis = 0).squeeze()

    losses_by_cluster = {"Clusters": test_clusters,
                    "Loss function": loss_fun,
                    "DCRNN-TL": avg_losses_model, 
                    "DCRNN": avg_losses_dcrnn,
                    "Average": avg_losses_avg, 
                    "Previous": avg_losses_previous, 
                    "VAR": avg_losses_var}
    
    return losses_by_timestamp, losses_by_cluster



#loss_functions = [masked_mse_np, masked_mape_np, masked_mae_np, masked_rmse_np]
loss_functions = [masked_mape_np, masked_mae_np, masked_rmse_np]
#loss_functions = [masked_mae_np]

for loss_fun in loss_functions:
    losses_by_timestamp, losses_by_cluster = compute_losses(loss_fun)
    message_info = f'Computed {losses_by_timestamp["Loss function"].__name__} over clusters: {losses_by_timestamp["Clusters"]}'
    print(message_info)
    logging.info(message_info)
    for time_stamp in range(horizon):
        message = f'Time stamp {time_stamp}: DCRNN-TL: {losses_by_timestamp["DCRNN-TL"][time_stamp]:.4f}, DCRNN: {losses_by_timestamp["DCRNN"][time_stamp]:.4f}, Average: {losses_by_timestamp["Average"][time_stamp]:.4f},   Previous: {losses_by_timestamp["Previous"][time_stamp]:.4f},  VAR: {losses_by_timestamp["VAR"][time_stamp]:.4f}' 
        print(message)
        logging.info(message)

    losses_time_df = pd.DataFrame({key: value for key, value in losses_by_timestamp.items() if key not in ['Clusters', 'Loss function']})
  #  losses_time_df.columns = ['Model', 'VAR', 'Previous', 'Average']
    losses_time_df['clusters'] = str(losses_by_timestamp['Clusters'])
    losses_time_df['metric'] = loss_fun.__name__



    losses_cluster_df = pd.DataFrame({key: (value if isinstance(value, np.ndarray) else [value]) for key, value in losses_by_cluster.items() if key not in ['Clusters', 'Loss function']})
    losses_cluster_df["cluster"] = losses_by_cluster['Clusters']

    # write to csv
    losses_time_df.to_csv(f'outputs/losses_by_timestamp_{loss_fun.__name__}_{output_name}.csv')
    losses_cluster_df.to_csv(f'outputs/losses_by_cluster_{loss_fun.__name__}_{output_name}.csv')

    # plot the losses over timestamps
    plt.plot(losses_by_timestamp["DCRNN-TL"], label = "DCRNN-TL")
    plt.plot(losses_by_timestamp["DCRNN"], label = "DCRNN")
    plt.plot(losses_by_timestamp["Average"], label = "Average")
    plt.plot(losses_by_timestamp["VAR"], label = "VAR")
    plt.plot(losses_by_timestamp["Previous"], label = "Previous")
    plt.title('Loss for DCRNN and Baselines over timestamps forecasted')
    plt.xlabel("Number of timestamps ahead")
    plt.ylabel(f'{losses_by_timestamp["Loss function"].__name__}')
    plt.legend()
    plt.savefig(f'plots/baseline_losses_{loss_fun.__name__}_{output_name}.png', format='png', dpi=300)
    plt.show()

    # plot the losses per cluster
    plt.scatter(losses_by_cluster["DCRNN-TL"], losses_by_cluster["VAR"])

    plt.title('Loss for DCRNN and Baselines over timestamps forecasted')
    plt.xlabel(f'{losses_by_cluster["Loss function"].__name__} DCRNN')
    plt.ylabel(f'{losses_by_cluster["Loss function"].__name__} VAR')
    plt.title("Loss for DCRNN and VAR for different clusters")
    plt.savefig(f'plots/losses_by_cluster_{loss_fun.__name__}_{output_name}.png', format='png', dpi=300)
    plt.show()



# also plot the model forecasted values and the true values
# we want to plot for the best cluster and the worst cluster

# retrieve the best and worst preforming cluster
min_index = losses_by_cluster["DCRNN-TL"].argmin() # lowest loss, best cluster
max_index = losses_by_cluster["DCRNN-TL"].argmax() # highest loss, worst cluster

best_cluster = losses_by_cluster["Clusters"][min_index]
worse_cluster = losses_by_cluster["Clusters"][max_index]

print("Best cluster", best_cluster)
print("Worst cluster", worse_cluster)
logging.info(f"Best cluster {best_cluster}")
logging.info(f"Worst cluster {worse_cluster}")













