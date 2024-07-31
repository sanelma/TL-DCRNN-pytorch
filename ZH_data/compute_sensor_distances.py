####### Script to compute distances between each pair of sensors in matched loop data
## Inputs:
# matchedloop csv files for Zurich and Luzern
# sumo net net.xml files for Zurich and Luzern
## Output:
# sensor_distances_luzern_zurich.h5 file that contains sumo road net distance, measured in travel time in seconds, between each pair of sensors



import numpy as np
import pandas as pd

from routing_lib.routing_utils import from_sumo_to_igraph_network, get_shortest_path
import os
import sys
import sumolib
import logging

# set sumo path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))


# configure logging
logging.basicConfig(filename=f"computeSensorDistances_both.log", level=logging.INFO, filemode='w', 
                        format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging is configured.")




# read in csv with the matched loop data
sensors_luzern = pd.read_csv('matchedloop_luzern.csv')
sensors_zh = pd.read_csv('matchedloop.csv')

# read in the sumo road network 
net_luzern = sumolib.net.readNet('LUZ.net.xml')
net_zh = sumolib.net.readNet('CZHAST.net.xml')



# one row detector. edge_id has edge id in the sumo network and lane id in the sumo network. 
# so we would take the edge id from the edge_id column for each pair and use shortest_path function



# igraph network with the zurich network
igraph_net_luzern = from_sumo_to_igraph_network(net_luzern)
igraph_net_zurich = from_sumo_to_igraph_network(net_zh)

# want to compute distances betweeen all pairs of sensors
# sensors has one row per detector. edge_id has edge id in the sumo network and lane id in the sumo network. 
# so we would take the edge id from the edge_id column for each pair and use shortest_path function

# filter out astra sensors - for Zurich, not Luzern
num_total_sensors_zh = len(sensors_zh)
astra_sensors = sensors_zh[sensors_zh['detid'].str.startswith('astra')]
sensors_city = sensors_zh[sensors_zh['detid'].str.startswith('K')]


assert len(astra_sensors) + len(sensors_city) == num_total_sensors_zh


sensors_zh = sensors_city


# get all unique edge ids
sensor_ids_zh = sensors_zh['detid'].unique()
sensor_ids_luzern = sensors_luzern['detid'].unique()

def compute_distances(sensor_ids, sensors, igraph_net):
    # create empty dataframe that will store the distances
    sensor_distances = pd.DataFrame(columns=['from', 'to', 'distance'])


    for n, from_sensor in enumerate(sensor_ids):
        logging.info(f"Progress ({n}/{len(sensor_ids)}")
        for to_sensor in sensor_ids:
            from_edge = sensors[sensors['detid'] == from_sensor]['edge_id'].values[0]
            to_edge = sensors[sensors['detid'] == to_sensor]['edge_id'].values[0]
            # as in the CA data, if the from and to edges are the same, the distance is 0 (lanes might be different)
            if from_edge == to_edge:
                new_row = pd.DataFrame({'from': [from_sensor], 'to': [to_sensor], 'distance': [0]})
                sensor_distances = pd.concat([sensor_distances, new_row], ignore_index=True)
                continue
            shortest_path_dict = get_shortest_path(igraph_net, from_edge, to_edge, 'traveltime')
            new_row = pd.DataFrame({'from': [from_sensor], 'to': [to_sensor], 'distance': [shortest_path_dict['cost']]})
            sensor_distances = pd.concat([sensor_distances, new_row], ignore_index=True)
    return sensor_distances

distances_zh = compute_distances(sensor_ids_zh, sensors_zh, igraph_net_zurich)
distances_luzern = compute_distances(sensor_ids_luzern, sensors_luzern, igraph_net_luzern)

logging.info("Generated sensor distances")


distances = pd.concat([distances_zh, distances_luzern])

# save the distances to a hdf5 file
distances.to_hdf('ZH_data/sensor_distances_luzern_zurich.h5', key='sensor_distances', mode='w')

print("Done")


