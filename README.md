# VODE Benchmark



This repository provides a benchmark to evaluate deep learning based visual odometry (VO) and monocular depth estimation (MDE). This work was inspired by [DeepVoFeat](https://github.com/Huangying-Zhan/Depth-VO-Feat), [SfmLearner](https://github.com/tinghuiz/SfMLearner) and [GeoNet](https://github.com/yzcjtr/GeoNet) and inherited source codes from them, especially from GeoNet.  Here are tools and datasets to evaluate the three models and compare them with [ORB SLAM](https://github.com/raulmur/ORB_SLAM). If you are interested in VO & MDE with deep learning, fork this repo!

## 1. Requirement

This work was developed in or with 
- Ubuntu 18.04 (but pretty sure that it works with Ubuntu 16.04)
- Python 3.6
- Tensorflow 1.12.0 (gpu version)
- CUDA 9.0 

To use this repo, just type below in terminal
```
pip install pipenv
git clone https://github.com/goodgodgd/vode-bench.git
cd vode-bench
pipenv install
pipenv shell
```
This repo manages python packages by pipenv. It is a better tool to setup the same environment in other PCs compared with virtualenv.   
First of all you have to install pip and then `pipenv install` to create virtual environment and install dependency packages. To activate the environment, `pipenv shell`.

## 2. Dataset

First click [here]() to start download, and read below. (sorry, now uploading...)   
The three previous works, [DeepVoFeat](https://github.com/Huangying-Zhan/Depth-VO-Feat), [SfmLearner](https://github.com/tinghuiz/SfMLearner) and [GeoNet](https://github.com/yzcjtr/GeoNet), were evaluated with [KITTI dataset](http://www.cvlibs.net/datasets/kitti/).  
So the READMEs in their repositories recommend visitor to download `KITTI raw dataset` and `KITTI odometry dataset`. However, They are huge datasets that takes more than 200GB. It may take a couple of days to download. After download, you have to customize the dataset to feed their network, resizing and concatenating images for train/val/test splits.  

Here, we provide ready-to-use dataset in `tfrecords` format. The structure and usage of `tfrecords` files refer to `data/tfrecord_feeder.py`.  

Other than tfrecords, DEVO dataset has all we need to study deep learning based VO and MDE. Let me introduce each directories.

- srcdata: The previous works rearranged KITTI dataset in almost the same way. This dir has the rearranged data extracted from `KITTI dataset`. The subdir `kitti_raw_eigen` is from the `KITTI raw dataset`, consists of horizontally concatenated images of three consecutive frames, and divide splits in the [Eigen split](https://cs.nyu.edu/~deigen/depth/) way. The other subdir `kitti_odom` is from the `KITTI odometry dataset` and consists of horizontally concatenated images of five consecutive frames.
- tfrecords: The dataset in`srcdata` was converted to `tfrecords` format in this dir. You can start from the ready-made tfrecord files.
- ckpts: This dir includes the checkpoints of `GeoNet` and `SfmLearner`. Each has depthnet and posenet for MDE and VO respectively.
- predicts: It has inference results from the previous works along with ground truth data. `orb_full` and `orb_shor` results were from [SfmLearner](https://github.com/tinghuiz/SfMLearner). `geonet` is a result of directly inferencing from the checkpoint while `deepvofeat` is provided data from its repository. SfmLearner has both: resulting data (`sfmlearner_data`) and inference result (`sfmlearner_pred`)
- evaluation: This repo has tools to evaluate various performance indices of VO and MDE. After evaluation, the result will be written in csv format into this dir.


## 3. Tools

To use funtionalities of this benchmark, `./devo_executer.sh` is almost all you need.
```
cd path/to/repo
pipenv shell
./devo_executer.sh [options]
```
To see what options there are, `./devo_executer.sh --help`.  
Before running the options, you have to set path variables in the script.
- RAW_DATA_ROOT: where "kitti_raw_data"(KITTI raw dataset) and "kitti_odom"(KITTI odometry dataset) dirs exist
- OUTPUT_PATH: the dataset dir downloaded and extracted in 2.
- MODEL_NAME: model name, upto now, the valid options are "geonet" or "sfmlearner"
