

from dcrnn_pytorch.dcrnn_supervisor import DCRNNSupervisor
#from dcrnn_pytorch.dcrnn_supervisor_old import DCRNNSupervisor
import yaml
import os
import logging
import datetime
import numpy as np

# read in parameters from config.yaml
with open("config_ch.yaml") as f:
    supervisor_config = yaml.safe_load(f)

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
script_directory = os.path.dirname(os.path.abspath(__file__))

# configure logging
logging.basicConfig(filename=f"logs/script_ch{formatted_time}.log", level=logging.INFO, filemode='w', 
                        format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging is configured.")


# train model
supervisor = DCRNNSupervisor(logging, **supervisor_config, script_directory = script_directory)
supervisor.train()

# evaluate model
test_loss, outputs = supervisor.evaluate(dataset="test")

# save outputs
np.save(f'outputs/outputs{formatted_time}.npy', outputs)
message = "Predictions saved"
print(message)
logging.info(message)

