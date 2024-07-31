# TL-DCRNN-pytorch
 Python implementation of TL-DCRNN 

Transfer Learning Diffusion Convolution Recurrent Neural Network (TL-DCRNN) is a transfer learning implementation of DCRNN which can be trained on one region of a highway network to forecast traffic in a different region of highway network where only realtime, but no historic data, is available. We here provide a PyTorch implementation fo TL-DCRNN, which was originally proposed by [Mallick et al in 2021](https://arxiv.org/abs/2004.08038) and is publicly available (in Tensorflow) on GitHub under [TL-DCRNN](https://github.com/tanwimallick/TL-DCRNN/tree/master). This code also builds upon the PyTorch implementation of DCRNN, publicly available under [DCRNN_PyTorch](https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/README.md).

The model was translated from Tensorflow 1 by [Sanelma Heinonen](https://www.linkedin.com/in/sanelma-heinonen/) as part of a masters thesis in Statistics at ETH Zürich and in cooperation with the start-up [Transcality](https://transcality.com/).

### Requirements
- python=3.12.2
- scipy=1.12.0
- numpy=1.26.
- pandas=2.2.1
- torch=2.2.0

### Data
TL-DCRNN requires the following inputs: 
- **Sensor data:** Location (latitude/longitude) of each traffic sensor. One row per traffic sensor. Columns: 'sensor_id', 'latitude', 'longitude'. 
- **Distance data:** The road network distances between each pair of sensors. One row per pair of sensor ids. Columns: 'from', 'to', 'distance'. 'from' and 'to' contain sensor ids of the corresponding sensors
- **Graph partition data:** Partition assigning traffic sensors to graph sub-clusters. List of length number sensors, corresponding to the subclusters of sensors in the same order as sensors in 'sensor data.'
- **Traffic data:** Time series data with the traffic measurements for each sensor. Column names are the sensor ids, one row per timestamp.

Precleaned data for the California highway network can be downloaded from the links under the GitHub for the original [TL-DCRNN](https://github.com/tanwimallick/TL-DCRNN/tree/master). The model can also be run on any traffic sensor data which follows the formats specified above. All input data should be kept in one folder. The name of the folder containing the input data and the names of the input files are to be specified in the `config.yaml` file.

#### Data preparation
The TL-DCRNN PyTorch model was created to test TL-DCRNN in Swiss cities. The data from Swiss cities is not publicly available. The following scripts which were used to prepare the data for TL-DCRNN are available in the `ZH_data` folder:
- `combine_zh_lz_counts.py`: filter and clean traffic data
- 'compute_sensor_distances.py': computes distances between each pair of sensors from [SUMO](https://eclipse.dev/sumo/) network data
- `generate_partition.py`: generates partition of subgraphs using Metis k-way partitioning algorithm

The data preparation scripts require the following dependencies: 
- networkx=3.2.1
- metis=0.2a5
- sumolib=1.20.0

### Training the model

The script `dcrnn_train.py` trains the model. The folder `dcrnn_pytorch` contains all files needed for training. All hyperparameters and input paths can be specified in `config.yaml`. The name of the config file used should be correctly specified in `dcrnn_train.py`. 

### Additional scripts 

- `evaluate_baselines.py` is a script for evaluating and comparing baseline models, given the outputs from `dcrnn_train.py`
- `create_dcrnn_plots.py` plots predictions, given the outputs from `dcrnn_train.py`
- `generate_descriptives_ch.py` generates descriptive statistics for Swiss data (not publicly available)








