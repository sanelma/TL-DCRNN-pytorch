# TL-DCRNN-pytorch
 Python implementation of TL-DCRNN 

 TL-DCRNN is a transfer learning implementation of DCRNN which can be trained on one region of a highway network to forecast traffic in a different region of highway network in which no historic data is available for training. TL-DCRNN was originally proposed by Mallick et al in 2021. The model is written in Tensorflow 1 and is publicly available on Github: https://github.com/tanwimallick/TL-DCRNN/tree/master. The code also builds upon the PyTorch implementation of DCRNN, publicly available on GitHub: https://github.com/xlwang233/pytorch-DCRNN

We here provide a PyTorch implementation of TL-DCRNN. Data for training can be downloaded from https://github.com/tanwimallick/TL-DCRNN/tree/master, or data in a similar format can be used. Data is to be kept in one folder, the name of which can be specified in the config.yaml file.

dcrnn_train.py trains the model using the relevant files contained the dcrnn_pytorch folder.

evaluate_baselines.py is a script for evaluating and comparing baseline models, given the outputs from dcrnn_train.py

create_dcrnn_plots.py plots predictions, given the outputs from dcrnn_train.py

generate_descriptives_ch.py generates descriptive statistics about Swiss data (not publicly available)

All hyperparameters and input paths can be specified in config.yaml

The following model versions were used:
[Update]
Python
Pytorch

The model was translated from Tensorflow 1 by Sanelma Heinonen as part of a masters thesis in Statistics at ETH Zürich and in cooperation with the start-up Transcality.



