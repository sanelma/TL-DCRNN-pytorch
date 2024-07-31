import os
import time
import sys

import numpy as np
import torch
import pandas as pd
#from torch.utils.tensorboard import SummaryWriter

#from lib import utils
try:
    from dcrnn_pytorch.dcrnn_model import DCRNNModel
    from dcrnn_pytorch.dcrnn_utils import masked_mae_loss, load_dataset, calculate_random_walk_matrix
except:
    from dcrnn_pytorch.dcrnn_model import DCRNNModel
    from dcrnn_pytorch.dcrnn_utils import masked_mae_loss, load_dataset, calculate_random_walk_matrix

    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNSupervisor:
    def __init__(self, logging = None, adj_mx = None, script_directory = None, **kwargs): # to-do: add adj_mx
        
        print("Device:", device)
        # directories
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._data_dir = kwargs.get('data_dir')



        self.script_directory = script_directory


        self.quick_run = True

        # data
        self.distances_path = os.path.join(self.script_directory, self._data_dir, kwargs.get('distances_file_name'))
        try:
            self.distances = pd.read_csv(self.distances_path)
        except:
            self.distances = pd.read_hdf(self.distances_path)


        self.sensors_path = os.path.join(self.script_directory, self._data_dir, kwargs.get('sensors_file_name'))
        self.sensors = pd.read_csv(self.sensors_path)

        self.speed_path = os.path.join(self.script_directory, self._data_dir, kwargs.get('traffic_file_name'))
        self.speed_data = pd.read_hdf(self.speed_path)

        try:
            self.speed_data_target = pd.read_hdf(os.path.join(self.script_directory, self._data_dir, kwargs.get('target_traffic_file_name')))
            self.speed_data = pd.concat([self.speed_data, self.speed_data_target], axis=1) 
            self.speed_data.index = pd.to_datetime(self.speed_data.index)
        except:
            pass


        self.partition_path = os.path.join(self.script_directory, self._data_dir, kwargs.get('partition_file_name'))
        self.partition = np.genfromtxt(self.partition_path, dtype=int, delimiter="\n", unpack=False)

        self.sensors['subgraph'] = self.partition 

        self.loop_ids = list(self.speed_data.columns)
        try: 
            self.loop_ids = [int(x) for x in self.loop_ids]
        except: 
            self.loop_ids = self.loop_ids

        self.sensors = self.sensors[self.sensors['sensor_id'].isin(self.loop_ids)]

        # then also keep only the first num_nodes sensors from each subgraph
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.sensors = self.sensors.groupby('subgraph').head(self.num_nodes)
        self.partition = self.sensors['subgraph'].values
        

        # parameters
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)


        log_level = self._kwargs.get('log_level', 'INFO')

        self._logger = logging

        self._logger.info(self._kwargs)
        print("Config:", self._kwargs)


        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
        self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder
        self.train_clusters = self._model_kwargs.get('train_clusters')
        self.test_clusters = self._model_kwargs.get('test_clusters')
        self.num_epochs = int(self._train_kwargs.get('num_epochs'))
        self.epsilon = float(self._train_kwargs.get('epsilon'))

        self.batch_size = int(self._model_kwargs.get('batch_size', 1))


        # setup model
        dcrnn_model = DCRNNModel(self._logger, **self._model_kwargs)
        self.dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model
        self._logger.info("Model created")



    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, data = None, train_cluster = None, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            # getting cluster information and calculating if doesn't exist
            if train_cluster != None:
                sclusters = [train_cluster]
                try:
                    adj_mx = self.adj_mx
                    sensor_ids = self.sensor_ids
                    sparse_rw_adj_mx = self.sparse_rw_adj_mx
                except:
                    print("Should only be passing in a cluster from when evaluating within train, when we also have adjacency matrix")
                    sys.exit(1)
            # if no traininig cluster passed in, we set cluster to test on to self.test_clusters
            else:
                sclusters = self.test_clusters

            self.dcrnn_model = self.dcrnn_model.eval()

            y_preds_all = {}
            y_truths_all = {}
            mean_loss = {}


            for cluster in sclusters:
                losses = []

                y_truths = []
                y_preds = []

                # if no cluster was passed in, need to get the graph info
                if train_cluster == None: 
                    print("Evaluating cluster: ", cluster)
                    self._logger.info("Evaluating cluster {}".format(cluster))
                    node_count, sensor_ids, adj_mx = self.cluster_data(cluster, self.partition, self.sensors, self.distances)

                    rw_adj_mx = calculate_random_walk_matrix(adj_mx).T
                    sparse_rw_adj_mx = self._build_sparse_matrix(rw_adj_mx)
                    speed_data = self.speed_data[sensor_ids]

                    data = load_dataset(speed_data, batch_size=self.batch_size, device = device, seq_len=self.seq_len, 
                                      horizon=self.horizon, quick_run= self.quick_run, 
                                      **self._data_kwargs
                                      )
                    # self._data is a dictionary with following keys: ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test', 'train_loader', 'val_loader', 'test_loader', 'scaler'] 
                    # x_train shape (num_train, seq_len, num_nodes, num_features)
                    # Dataloaders have attributes xs and ys which are numpy arrays of shape (num_samples, seq_len, num_nodes, num_features) and (num_samples, horizon, num_nodes, num_features) respectively
                    # in the data loaders, num_samples may be larger, because they can be padded in order to be divisible by batch_size
                    # in addition, data loader data for train is shuffled
                    self.standard_scaler = data['scaler']

                val_iterator = data['{}_loader'.format(dataset)].get_iterator()
            
                for _, (x, y) in enumerate(val_iterator):

                    x, y = self._prepare_data(x, y) # this just does some reshaping to  (seq_len, batch_size, num_sensor * input_dim)

                    output = self.dcrnn_model(x, adj_mx = sparse_rw_adj_mx) # output shape (horizon, batch_size, num_nodes * output_dim)
                    loss = self._compute_loss(y, output)
                    losses.append(loss.item())

                    y_truths.append(y.cpu()) # y_preds is a list of length batches_per_epoch, where each element is a tensor of shape (horizon, batch_size, num_nodes * output_dim)
                    y_preds.append(output.cpu())

                
                y_preds = np.concatenate(y_preds, axis=1) # (horizon, epoch_size, num_nodes)
                y_truths = np.concatenate(y_truths, axis=1) 

                y_truths_scaled = []
                y_preds_scaled = []
                for t in range(y_preds.shape[0]):
                    y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                    y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                    y_truths_scaled.append(y_truth)
                    y_preds_scaled.append(y_pred)
                
                # y_truths_scaled is a list of length horizon, where each element is a numpy array of shape (epoch_size, num_nodes)
                # y_preds_all is a dictionary, where keys are the clustesr ids. items are lists of length horizon, where each item is np array (epoch size, num nodes)
                # mean_loss is a dictionary, keys are the cluster ids, items are the mean loss for that cluster
                y_preds_all[cluster] = y_preds_scaled 
                y_truths_all[cluster] = y_truths_scaled
                mean_loss[cluster] = np.mean(losses)
            
            return mean_loss, {'prediction': y_preds_all, 'truth': y_truths_all}

    def _train(self, base_lr, 
               steps, patience=50, num_epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10,  **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=self.epsilon)

    #    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
    #                                                        gamma=lr_decay_ratio)

        self._logger.info('Start training ...')
        self._logger.info('Constant learning rate. no schedule. adam')
        print('Constant learning rate. no schedule. adam')  



        batches_seen = 0
        epochs_seen = 0

        for cluster in self.train_clusters:
            print("Training cluster: ", cluster)
            self._logger.info("Training cluster {}".format(cluster))
            node_count, sensor_ids, adj_mx = self.cluster_data(cluster, self.partition, self.sensors, self.distances)
            self._logger.info("Cluster data loaded")


            speed_data = self.speed_data[sensor_ids]
            speed_data = speed_data.dropna()
            # set attributes so they can also be used for validation 
            self.adj_mx = adj_mx
            self.sensor_ids = sensor_ids

            # sparse random walk matrix 
            self.rw_adj_mx = calculate_random_walk_matrix(self.adj_mx).T
            self.sparse_rw_adj_mx = self._build_sparse_matrix(L = self.rw_adj_mx)

            # data set - normalization occurs in here. so, each cluster is normalized separately
            data = load_dataset(speed_data, batch_size=self.batch_size, device = device, seq_len=self.seq_len, 
                                      horizon=self.horizon, quick_run= self.quick_run, 
                                      **self._data_kwargs
                                      )
            # self._data is a dictionary with following keys: ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test', 'train_loader', 'val_loader', 'test_loader', 'scaler'] 
            # x_train shape (num_train, seq_len, num_nodes, num_features)
            # Dataloaders have attributes xs and ys which are numpy arrays of shape (num_samples, seq_len, num_nodes, num_features) and (num_samples, horizon, num_nodes, num_features) respectively
            # in the data loaders, num_samples may be larger, because they can be padded in order to be divisible by batch_size
            # in addition, data loader data for train is shuffled
            self.standard_scaler = data['scaler']
            epochs_per_cluster = self.num_epochs // len(self.train_clusters)

            for epoch_num in range(epochs_per_cluster):

                self.dcrnn_model = self.dcrnn_model.train()

           
                train_iterator = data['train_loader'].get_iterator()
                losses = []

                start_time = time.time()

                for _, (x, y) in enumerate(train_iterator):
                    # x.shape is (batch_size, seq_len, num_nodes, num_features)
                    # y.shape (batch_size, horizon, num_nodes, num_features)
                    optimizer.zero_grad()
                    # note: each time it runs this, it sends x and y to device. maybe this is why so slow
                    x, y = self._prepare_data(x, y) # x: torch.Size([seq_len, batch_size, num_nodes * input_features])

                    output = self.dcrnn_model(x = x,
                                              adj_mx = self.sparse_rw_adj_mx,
                                               labels =  y,
                                                batches_seen = batches_seen)

                    if batches_seen == 0:
                        # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=self.epsilon)

                    loss = self._compute_loss(y, output)

                    self._logger.debug(loss.item())

                    losses.append(loss.item())

                    batches_seen += 1
                    loss.backward()

                    # gradient clipping - this does it in place
                    torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)

                    optimizer.step()
                self._logger.info("epoch complete")
             #   lr_scheduler.step()
                self._logger.info("evaluating now!")

                val_loss, _ = self.evaluate(data, train_cluster  = cluster, dataset='val', batches_seen=batches_seen)
                val_loss = val_loss[cluster]

                end_time = time.time()



                if (epoch_num % log_every) == log_every - 1:
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                            '{:.1f}s'.format(epochs_seen, num_epochs, batches_seen,
                                             np.mean(losses), val_loss, base_lr,
                                    #        np.mean(losses), val_loss, lr_scheduler.get_last_lr()[0],
                                            (end_time - start_time))
                   

                if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                    test_loss, _ = self.evaluate(data, train_cluster = cluster, dataset='test', batches_seen=batches_seen)
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                            '{:.1f}s'.format(epochs_seen, num_epochs, batches_seen,
                                             np.mean(losses), test_loss[cluster], base_lr,
                                        #    np.mean(losses), test_loss, lr_scheduler.get_last_lr()[0],
                                            (end_time - start_time))
                self._logger.info(message)
                print(message)

                if val_loss < min_val_loss:
                    wait = 0
                    if save_model:
                        model_file_name = self.save_model(epochs_seen)
                        self._logger.info(
                            'Val loss decrease from {:.4f} to {:.4f}, '
                            'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                    min_val_loss = val_loss
         

                epochs_seen += 1
        model_file_name = self.save_model(epochs_seen)
        

    def _prepare_data(self, x, y):
        # change from (batch_size, seq_len, num_sensor, input_dim) to (seq_len, batch_size, num_sensor, input_dim)
        x = x.permute(1, 0, 2, 3) 
        y = y.permute(1, 0, 2, 3)

        # next change from (seq_len, batch_size, num_sensor, input_dim) to (seq_len, batch_size, num_sensor * input_dim)
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)

        return x, y



    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
    
        ############### Cluster data ####################

    def get_adjacency_matrix(self, distance_df, sensor_ids, normalized_k=0.1):
    
        """ 
        :param distance_df: data frame with three columns: [from, to, distance].
        :param sensor_ids: list of sensor ids in the cluster
        :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
        Returns: adjacency matrix for the subgraph cluster. This is the essentially the distance matrix with a Gaussian kernel applied, and entries below a threshold are set to zero for sparsity
        """

        num_sensors = len(sensor_ids)
        dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
        dist_mx[:] = np.inf
            
        # getting index of sensor_ids
        sensor_id_to_ind = {}
        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i
        
        # Building dist_mx which has distances between sensors that are in the subgraph
        # only cares about distances between sensors included in sensor_ids list – building matrix for a subgraph
        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
        
        # getting all non-infinite distances and computing their standard deviation
        distances = dist_mx[~np.isinf(dist_mx)].flatten() # ~ is a "not"
        std = distances.std()
        adj_mx = np.exp(-np.square(dist_mx / std)) # transforming distance matrix by applying gaussian kernel
        
        adj_mx[adj_mx < normalized_k] = 0

        return adj_mx
    

    def _build_sparse_matrix(self, L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L


    def cluster_data(self, cluster, partition, sensors, distances):
        """ 
        :param cluster: 'cluster' is just the cluster ID number
        :param partition:
        :param sensors: 
        :param distances: 
        Returns: node_count, sensor_ids, adj_mx for the cluster
        """
        indices = partition==cluster # get the indices of the sensors that belong to the cluster of interest
        part_df = sensors[indices] # lat/lon 
        distance_df = distances.loc[(distances['from'].isin(part_df['sensor_id'])) & (distances['to'].isin(part_df['sensor_id']))]
        distance_df = distance_df.reset_index(drop=True)
        distance_df['from'] = distance_df['from'].astype('str')
        distance_df['to'] = distance_df['to'].astype('str')
        sensor_ids = part_df['sensor_id'].astype(str).values.tolist()

        node_count = len(sensor_ids)       
        adj_mx = self.get_adjacency_matrix(distance_df, sensor_ids) 

        return node_count, sensor_ids, adj_mx
    
    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch