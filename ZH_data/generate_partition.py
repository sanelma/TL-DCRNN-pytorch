##### Script to filter, clean, and process the traffic count data
## Inputs: 
# sensor_distances.h5
# zurich_counts.h5
# luzern_jan_apr2015.csv 
## Output:
# luzern_and_zurich_counts.h5, luzern_counts.h5, zurich_counts.h5


import numpy as np
import pandas as pd
import glob

import os
import sys
import sumolib
import igraph
from routing_lib.routing_utils import from_sumo_to_igraph_network, get_shortest_path

import metis
import networkx as nx

import folium
import branca.colormap as cm
from matplotlib.cm import get_cmap

### 
n_subgraphs = 13


# read hdf
distances = pd.read_hdf('zh_data/sensor_distances_luzern_zurich.h5')

counts_zurich = pd.read_hdf('zh_data/zurich_counts.h5')
counts_luzern = pd.read_hdf('zh_data/luzern_counts.h5')


# sensor coordinates
sensor_coords = pd.read_csv('ZH_data/detector_coordinates_zurich.csv')
sensor_coords_luzern = pd.read_csv('ZH_data/detector_coordinates_luzern.csv')

# ids that are in the counts data
sensor_ids = counts_zurich.columns
sensor_ids_luzern = counts_luzern.columns


sensor_coords = sensor_coords[sensor_coords['id'].isin(sensor_ids)]
sensor_coords_luzern = sensor_coords_luzern[sensor_coords_luzern['id'].isin(sensor_ids_luzern)]
distances = distances[distances['from'].isin(sensor_ids) & distances['to'].isin(sensor_ids)]




# add coordinates for convenience
distances['from_lat'] = distances['from'].map(sensor_coords.set_index('id')['latitude'])
distances['from_lon'] = distances['from'].map(sensor_coords.set_index('id')['longitude'])
distances['to_lat'] = distances['to'].map(sensor_coords.set_index('id')['latitude'])
distances['to_lon'] = distances['to'].map(sensor_coords.set_index('id')['longitude'])

# distances that are less than 90 seconds – will use these as adjacent nodes for the graph
distances_filtered = distances[distances['distance']<90]
# create a list of tuples, where each tuple is (from, to) from the distances_filtered dataframe
edges = list(zip(distances_filtered['from'], distances_filtered['to']))


# create graph out of the sensors
G = nx.DiGraph(edges) 
partition = metis.part_graph(G, nparts=n_subgraphs, objtype   = 'cut')[1]  # 'cut' or 'vol'

# assert that each node gets assigned to a partition
assert len(G.nodes) == len(partition)


# create dataframe with the nodes and the partition
partition_df = pd.DataFrame({'node': list(G.nodes), 'partition': partition})

sensor_coords['partition'] = sensor_coords['id'].map(partition_df.set_index('node')['partition'])

partition = sensor_coords['partition'].tolist()

# add luzern sensors to the end of the partition as their own subgraph
partition = partition + [n_subgraphs] * len(counts_luzern.columns)

partition_str = "\n".join(map(str, partition))

sensor_coords_for_dcrnn = pd.DataFrame({'sensor_id': sensor_coords['id'],
                                         'latitude': sensor_coords['latitude'], 
                                         'longitude': sensor_coords['longitude']})

sensor_coords_for_dcrnn_luzern = pd.DataFrame({'sensor_id': sensor_coords_luzern['id'],
                                         'latitude': sensor_coords_luzern['latitude'], 
                                         'longitude': sensor_coords_luzern['longitude']})






sensor_coords_for_dcrnn = pd.concat([sensor_coords_for_dcrnn, sensor_coords_for_dcrnn_luzern])




# write sensor coordinates
sensor_coords_for_dcrnn.to_csv('ZH_data/sensor_coords_for_dcrnn.csv', index=False)


file_path = "ZH_data/partition.txt.part.64"  # Replace with your actual file path

with open(file_path, "w") as file:
    file.write(partition_str)