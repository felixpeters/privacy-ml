# privacy-ml

# Setup

0. If a CUDA GPU is available: Install Cuda Toolkit (https://developer.nvidia.com/cuda-downloads)
1. Open an anaconda prompt and navigate to project folder (privacy-ml).
2. conda env create --file requirements.txt --name seminar
3. conda activate seminar
4. pip install opacus syft=0.5.0rc1 xlrd "monai-weekly[all]"

# Experiments

Two different experiements can be conducted with this code. In the first run the dataset has to be downloaded, which is acoomplished by passing download=True to the Loader function.

1. Federated Learning with 1 Data Owner and 1 Datascientist
    -> Run both notebooks starting with "1DO_1DS" simultaneously.
    -> Maybe the sleep timer in the data scientists side needs to be adjusted so the training doesn´t start before the data is loaded into the duet store.

2. Ground truth of the model training
    -> This is already implemented as a grid search for different parameters
    -> Just run the notebook called "Ground_truth"
