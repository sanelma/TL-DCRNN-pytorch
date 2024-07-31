# Script generates descriptive statistics about the Zurich data

import yaml
import os
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib import dates as mdates
import sumolib

from lib.plotting import generate_weekday_weekend_averages, plot_averages_over_day


# sumo data – in order to split edges by road type for plotting
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# sumo network and loopids
net_zurich = sumolib.net.readNet('ZH_data/sumo_data/CZHAST.net.xml')
net_luzern = sumolib.net.readNet('ZH_data/LUZ.net.xml')
matchedloop_lucern = pd.read_csv('ZH_data/matchedloop_luzern.csv')
matchedloop_zurich = pd.read_csv('ZH_data/sumo_data/matchedloop.csv')


# read in parameters from config.yaml - will generate descriptives based on what is set in the current version of the config file
with open("config_ch.yaml") as f:
    supervisor_config = yaml.safe_load(f)

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

script_directory = ""

######## Read in data
data_kwargs = supervisor_config.get('data')
model_kwargs = supervisor_config.get('model')
data_dir = supervisor_config.get('data_dir')

# Read in data
distances_path = os.path.join(script_directory, data_dir, supervisor_config.get('distances_file_name'))
try:
    distances = pd.read_csv(distances_path)
except:
    distances = pd.read_hdf(distances_path)

sensors_path = os.path.join(script_directory, data_dir, supervisor_config.get('sensors_file_name'))
sensors = pd.read_csv(sensors_path)


speed_path = os.path.join(script_directory, data_dir, supervisor_config.get('traffic_file_name'))
speed_data = pd.read_hdf(speed_path)

# in case the target data is in a different file for a different time period

try:
    target_speed_path =  os.path.join(script_directory, data_dir, supervisor_config.get('target_traffic_file_name'))
    target_speed_data = pd.read_hdf(target_speed_path)
    speed_data = pd.concat([speed_data, target_speed_data], axis=1) 
 #   speed_data.index = pd.to_datetime(speed_data.index)
except:
    pass



partition_path = os.path.join(script_directory, data_dir, supervisor_config.get('partition_file_name'))
partition = np.genfromtxt(partition_path, dtype=int, delimiter="\n", unpack=False)

sensors['subgraph'] = partition 

loop_ids = list(speed_data.columns)
try: 
    loop_ids = [int(x) for x in loop_ids]
except: 
    loop_ids = loop_ids

sensors = sensors[sensors['sensor_id'].isin(loop_ids)]

# then also keep only the first num_nodes sensors from each subgraph
num_nodes = int(model_kwargs.get('num_nodes'))
sensors = sensors.groupby('subgraph').head(num_nodes)
partition = sensors['subgraph'].values



train_clusters = model_kwargs.get('train_clusters')
test_clusters = model_kwargs.get('test_clusters')
clusters_used = train_clusters + test_clusters
clusters_used = np.unique(clusters_used).tolist()

sensors = sensors[sensors['subgraph'].isin(clusters_used)]



# sensor ids of sensors that are used
sensor_ids = sensors['sensor_id'].values
sensor_ids = [str(x) for x in sensor_ids]

sensor_ids_source = sensors[sensors['subgraph'].isin(train_clusters)]['sensor_id'].values
sensor_ids_source = [str(x) for x in sensor_ids_source]
sensor_ids_target = sensors[sensors['subgraph'].isin(test_clusters)]['sensor_id'].values
sensor_ids_target = [str(x) for x in sensor_ids_target]

# filter speed data: only include the sensors that we are usuing, and the top n_samples_quick rows
num_samples_quick = int(data_kwargs.get('n_samples_quick'))
speed_data = speed_data[:num_samples_quick]
speed_data = speed_data.loc[:,sensor_ids]
speed_data.index = pd.to_datetime(speed_data.index)
target_speed_data.index = pd.to_datetime(target_speed_data.index)


seq_len = model_kwargs.get('seq_len')
horizon = model_kwargs.get('horizon')

print("data loaded")


## Get the means and standard deviations of the clusters that are used
means = {}
stds = {}
maxs = {}
mins = {}

for cluster in np.unique(train_clusters).tolist():
    # get sensor ids of the sensors in the cluster 
    sensors_in_cluster = sensors[sensors['subgraph'] == cluster]['sensor_id'].values
    speed_data_cluster = speed_data.loc[:,[str(x) for x in sensors_in_cluster]]

    speed_data_cluster.index = speed_data_cluster.index.astype(str)

    speed_data_cluster = speed_data_cluster.reset_index()
    speed_data_cluster = speed_data_cluster.melt(id_vars=['datetime'])


    speed_data_cluster['value'].mean()

    print(f"Mean of cluster {cluster}:",  speed_data_cluster['value'].mean())
    print(f"STD of cluster {cluster}:",  speed_data_cluster['value'].std())
    print(f"Max of cluster {cluster}:",  speed_data_cluster['value'].max())
    print(f"Min of cluster {cluster}:",  speed_data_cluster['value'].min())

    means[cluster] = speed_data_cluster['value'].mean()
    stds[cluster] = speed_data_cluster['value'].std()
    maxs[cluster] = speed_data_cluster['value'].max()
    mins[cluster] = speed_data_cluster['value'].min()

# add in target
target_speed_data.index = target_speed_data.index.astype(str)
temp = target_speed_data.reset_index()
temp = temp.melt(id_vars=['datetime'])
means['target'] = temp['value'].mean()
stds['target'] = temp['value'].std()
maxs['target'] = temp['value'].max()
mins['target'] = temp['value'].min()

# want to look at distribution of each cluster

clusters = [str(x) for x in list(means.keys())] 
mean_values = list(means.values())
std_values = list(stds.values())
min_values = list(mins.values())
max_values = list(maxs.values())



# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(clusters, mean_values, yerr = std_values, color = "grey")
plt.scatter(clusters, min_values, color='red', label='Min', zorder=5)
plt.scatter(clusters, max_values, color='blue', label='Max', zorder=5)
plt.title('Means, standard deviations, minimums, and maximums of counts by cluster')
plt.xlabel('Cluster')
plt.ylabel('Mean traffic count (vehicles per hour) ')
plt.savefig(f'plots/ch_counts_dists_by_cluster.png', format='png', dpi=300)  # Adjust the filename and format as needed



# write cluster summaries to csv
cluster_summaries = pd.DataFrame([means, stds, mins, maxs]).T
cluster_summaries.columns = ["Mean", "Std", "Min", "Max"]
cluster_summaries.to_csv(f"outputs/cluster_summaries_ch.csv")

# distribution speeds of all sensors
speed_data.index = speed_data.index.astype(str)

df = speed_data.reset_index()
df = df.melt(id_vars=['datetime'])


plt.figure(figsize=(10, 6))
plt.hist(df['value'], bins = 50, color = "grey")
plt.title('Distribution of traffic data counts')
plt.ylabel('Count')
plt.xlabel('Sensor reading (count of vehicles, vehicles/hour)')
plt.savefig(f'plots/distribution_of_traffic_speeds_ch.png', format='png', dpi=300)  # Adjust the filename and format as needed
#plt.show()





###### going to plot average speeds over the course of a day, averaged over all clusters



train_weekday, train_weekend = generate_weekday_weekend_averages(speed_data[sensor_ids_source])
target_weekday, target_weekend = generate_weekday_weekend_averages(target_speed_data)



plot_averages_over_day(train_weekday, target_weekday, 
                       day = "weekdays", 
                       measure = "Counts", 
                       source = "Zurich", 
                       target="Lucerne", 
                       save_plot=True)

plot_averages_over_day(train_weekend, target_weekend, 
                       day = "weekends", 
                       measure = "Counts", 
                       source = "Zurich", 
                       target="Lucerne", 
                       save_plot=True)




# want to also generate these plots for only segments on roads of type 'highway.primary'


sensors_lucern = sensors[sensors['subgraph'].isin(test_clusters)]
edge_types_luzern = []

# filter matched loop for only the sensors used
matchedloop_lucern = matchedloop_lucern[matchedloop_lucern['detid'].isin(sensor_ids_target)]
matchedloop_zurich = matchedloop_zurich[matchedloop_zurich['detid'].isin(sensor_ids_source)]

# getting the road types for luzern and zurich
edge_types_luzern = []
for edge_id in matchedloop_lucern['edge_id']:
    edge = net_luzern.getEdge(edge_id)
    type = edge.getType()
    edge_types_luzern.append(type)

edge_types_zurich = []
for edge_id in matchedloop_zurich['edge_id']:
    edge = net_zurich.getEdge(edge_id)
    type = edge.getType()
    edge_types_zurich.append(type)

edge_types = edge_types_zurich + edge_types_luzern

# add road type to the sensors dataframe
sensors['road_type'] = edge_types

# getting the sensor ids for source and target ids that are on highways
sensor_ids_zurich_highway = sensors[(sensors['subgraph'].isin(train_clusters)) & (sensors['road_type'] == 'highway.primary')]['sensor_id'].values
sensor_ids_luzern_highway = sensors[(sensors['subgraph'].isin(test_clusters)) & (sensors['road_type'] == 'highway.primary')]['sensor_id'].values



train_weekday, train_weekend = generate_weekday_weekend_averages(speed_data[sensor_ids_zurich_highway])
target_weekday, target_weekend = generate_weekday_weekend_averages(target_speed_data[sensor_ids_luzern_highway])

plot_averages_over_day(train_weekday, target_weekday, 
                       day = "weekdays", 
                       measure = "Counts", 
                       source = "Zurich - Primary highways", 
                       target="Lucerne - Primary highways", 
                       save_plot=True)

plot_averages_over_day(train_weekend, target_weekend, 
                       day = "weekends", 
                       measure = "Counts", 
                       source = "Zurich - Primary highways", 
                       target="Lucerne - Primary highways", 
                       save_plot=True)






