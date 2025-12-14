# A Wind-Guided Spatial-Temporal-Variable Fusion Network for predicting multiple air pollutants

This repo is the implementation of our manuscript entitled A Wind-Guided Spatial-Temporal-Variable Fusion Network for predicting multiple air pollutants. The code is based on Pytorch 1.12.1, and tested on a GeForce RTX 4090 GPU with 24GB memory.


## The Framework of WG-STVFNet

![WG-STVFNet](./Figure/WG-STVFNet.png)


## Requirements
MSTVFFN uses the following dependencies

- Pytorch 1.12.1 and its dependencies
- Numpy and Pandas
- CUDA 11.8 or latest version

## Dataset
- Beijing dataset: The Beijing multi-site air quality dataset is obtained from Beijing-Multi-Site-Air-Quality-Data-Data-Set (https://github.com/Afkerian/Beijing-Multi-Site-Air-Quality-Data-Data-Set/tree/main)

- London dataset: The London air quality dataset is obtained from KDD2018_FreshAir (https://github.com/B04902039/KDD2018_FreshAir)

The processed sample data provided in this repository are located in the [click here](./MSTVFFN/Data).

The Data provided in this repository are sample data, intended only to demonstrate the code workflow and data format specifications. The sample data are not sufficient for model training. For full experiments, please obtain the complete dataset following the instructions in the related paper or documentation, or prepare your own data in the same format.


## Folder Structure
We list the code of the major modules as follows:
- The main function to train/test our model: [click here](./WG-STVFNet/main.py)
- The source code of our model: [click here](./WG-STVFNet/model.py)
- Train and test data preporcessing are located at: [click here](./WG-STVFNet/Data_process.py)
- Metric computations: [click here](./WG-STVFNet/utils.py)

## Arguments
We introduce some major arguments of our main function here.

Training settings:
- train\_rate: rate of train set
- test\_rate: rate pf test set
- lag: time length of hidtorical steps
- pre\_len: time length of future steps
- num\_nodes: the number of stations
- batch\_size: training or testing batch size
- input\_dim: the feature dimension of inputs
- learning\_rate: the learning rate at the beginning
- epochs: training epochs
- early\_stop_patience: the patience of early stopping
- device: using which GPU to train our model
- seed: the random seed for experiments

## Citation
- If you find our work useful in your research, please cite: