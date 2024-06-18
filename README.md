# MTL-CSDNN
This repository provides the code for a deep learning model named MTL-CSDNN, which have the ability of 
precisely predicting mortality risk for multiple chronic diseases in the elderly using real-world data.
## Dependencies
- Python 3.9.17
- NumPy (currently tested on version  1.26.2)
- PyTorch (currently tested on version 2.0.1)
- scipy 1.11.4
- pytorch-widedeep 1.4.0
- pytorch-tabnet 4.1.0
- tqdm 4.65.0
- optuna 3.5.0
## How to use
Unfortunately, the data set of this repository is confidential for some reason, so this code cannot run either.
The architecture of the code is described next:
- config.py: changing global setting or params here
- data_loader.py: providing dataloader for the model
- MTL_CS_DNN_run.py: training MTL-CSDNN here, and also you can use k-folds and trials by changing the params
- net.py: coding the structure of the model
- pack_task.py: some baselines including: tabnet,catboost,xgboost
- trial_pytorch.py: some baselines including: global-dnn,logistic regression,1d-cnn,saint,tab transform
