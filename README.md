# VODE Benchmark

This repository provides a benchmark to evaluate deep learning based visual odometry (VO) and monocular depth estimation (MDE). This work was inspired by [DeepVoFeat](https://github.com/Huangying-Zhan/Depth-VO-Feat), [SfmLearner](https://github.com/tinghuiz/SfMLearner) and [GeoNet](https://github.com/yzcjtr/GeoNet) and inherited source codes from them, especially from GeoNet.  Here are tools and datasets to evaluate the three models and compare them with [ORB SLAM](https://github.com/raulmur/ORB_SLAM). If you are interested in VO & MDE with deep learning, fork this repo!

## 1. Requirement

This work was developed in or with 
- Ubuntu 18.04 (but pretty sure that it works with Ubuntu 16.04)
- Python 3.6
- Tensorflow 1.12.0 (gpu version)
- CUDA 9.0 

To use this repo, just type below in terminal
```bash
pip install pipenv
git clone https://github.com/goodgodgd/vode-bench.git
cd vode-bench
pipenv install
pipenv shell
```
This repo manages python packages by pipenv. It is a better tool to setup the same environment in other PCs compared with virtualenv.   
First of all you have to install pip and then `pipenv install` to create virtual environment and install dependency packages. To activate the environment, `pipenv shell`.

## 2. Dataset

First click [here](https://drive.google.com/open?id=1bXoNly2l0OTkhwdk7ADS0glwgx9Kgb-R) to start download, and read below.   
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
```bash
cd path/to/repo
pipenv shell
./devo_executer.sh [option]
```
The previous works let users to type commands with long~~~~ options like  (no offense)
```bash
python geonet_main.py --mode=train_rigid --dataset_dir=/path/to/formatted/data/ --checkpoint_dir=/path/to/save/ckpts/ --learning_rate=0.0002 --seq_length=3 --batch_size=4 --max_steps=350000 
```
Typing this long command was annoying to me, so I wrote a script `devo_executer.sh` using internal variables. As the options and variables are saved in the script, you don't have to type all the options every time you run something. I simplified options and the detailed options can be edited in the script.

To see what options there are, `./devo_executer.sh --help`.  
Before running the options, you have to set path variables in the script.
- RAW_DATA_ROOT: where "kitti_raw_data"(KITTI raw dataset) and "kitti_odom"(KITTI odometry dataset) dirs exist
- OUTPUT_PATH: the dataset dir downloaded and extracted in 2.
- MODEL_NAME: model name, upto now, the valid options are "geonet" or "sfmlearner"

If you 
1. setup and activated the environment
2. dowloaded and extracted the dataset, 
3. and set the path variables,  

then enjoy the benchmark.  

Here are available options, step by step
1. prepare_paths: make required dirs under `OUTPUT_PATH`
2. prepare_kitti_eigen: prepare stacked images and gt depth data from kitti raw dataset
3. prepare_kitti_odom: prepare stacked images and gt pose data from kitti odom dataset
4. make_tfrecord_eigen: create tfrecord files from the results of prepare_kitti_eigen
5. make_tfrecord_odom: create tfrecord files from the results of prepare_kitti_odom
6. train_rigid: train pose and depth prediction model
7. pred_depth: predict depths from test data and save them
8. pred_pose: predict poses from test data and save them. The predicted poses are in TUM dataset format and poses of only 5 frame sequences.
9. eval_depth: compute performance indices to evaluate depth prediction results
10. eval_pose: compute performance indices to evaluate pose prediction results
11. eval_traj: to evaluate trajectory estimation performance, reconstruct full trajectries from 5 frame sequence poses and evaluate the accuracy of trajectories.

Note that if the dataset in 2 includes all the results of the steps above. So you can start from any step.

## 4. Create New Model

`models` directory collects VO & MDE models. The codes are copied from the repositories of the original authors but adjusted to meet the rules for the new class.  
If you want to try a new model, create a new class in a new directory here and follow the rules below.  

- A new class **must** inherit from `ModelBase` in abstracts.py

- A new class **must** implement `ModelBase`'s abstract methods

  - build_model(): build graphs for NN models and computing losses

  - get_loss(): returns total loss tensor

  - get_pose_pred(): return pose prediction tensor

  - get_depth_pred(): return depth prediction tensor

## 5. Evaluation Results

Notations
- (d): prediction results provided from the authors
- (t): predictions results by executing the models from the provided checkpoints
- rel: relative or ratio error
- rms: root mean square
- te: translational error
- re: rotational error
- teNf: average translational error of N frame forward prediction. e.g. If the current frame index is 100, te3f means the average relative translational error of the frame 103
- reNf: translational error of N frame forward prediction. e.g. If the current frame index is 100, re3f means the relative rotational error of the frame 103
- interval: when reconstructing full trajectories from 5 frame trajectories, poses are accumulated for every `interval` -th frames 

Note that the ORB SLAM's results were from https://github.com/tinghuiz/SfMLearner  
I am planning to create ORB SLAM's results by myself  

### Depth evaluation result

| model          | abs_rel  | sq_rel   | rms      | log_rms  | delta<1.25 | delta<1.25^2 | delta<1.25^3 |
| -------------- | -------- | -------- | -------- | -------- | ---------- | ------------ | ------------ |
| deepvofeat (d) | 0.135618 | 1.137796 | 5.466103 | 0.215881 | 0.832680   | 0.941568     | 0.975176     |
| geonet (t)     | 0.185811 | 1.722285 | 6.438714 | 0.261068 | 0.734080   | 0.910604     | 0.965618     |
| sfmlearn (d)   | 0.197828 | 1.836316 | 6.564524 | 0.274969 | 0.717570   | 0.901032     | 0.960593     |
| sfmlearner (t) | 0.211769 | 1.850629 | 7.285805 | 0.301619 | 0.664518   | 0.872548     | 0.948744     |


### Pose evaluation result

| model          | drive | te_mean | te_std | re_mean | re_std | te1f   | te2f   | te3f   | te4f   |
|----------------|-------|---------|--------|---------|--------|--------|--------|--------|--------|
| deepvofeat (d) | 9     | 0.0311  | 0.0246 | 0.2414  | 0.2029 | 0.0129 | 0.0249 | 0.0370 | 0.0495 |
| deepvofeat (d) | 10    | 0.0297  | 0.0293 | 0.2522  | 0.2356 | 0.0130 | 0.0239 | 0.0350 | 0.0469 |
| geonet (t)     | 9     | 0.0271  | 0.0232 | 0.1953  | 0.1328 | 0.0113 | 0.0211 | 0.0318 | 0.0443 |
| geonet (t)     | 10    | 0.0266  | 0.0261 | 0.1931  | 0.1341 | 0.0117 | 0.0207 | 0.0311 | 0.0430 |
| orb_full (d)   | 9     | 0.0275  | 0.0199 | 0.0275  | 0.0144 | 0.0141 | 0.0196 | 0.0334 | 0.0363 |
| orb_full (d)   | 10    | 0.0240  | 0.0251 | 0.0328  | 0.0177 | 0.0090 | 0.0178 | 0.0254 | 0.0323 |
| orb_short (d)  | 9     | 0.0290  | 0.0557 | 0.0387  | 0.0937 | 0.0108 | 0.0243 | 0.0334 | 0.0460 |
| orb_short (d)  | 10    | 0.0376  | 0.1004 | 0.0629  | 0.2435 | 0.0137 | 0.0302 | 0.0428 | 0.0576 |
| sfmlearner (d) | 9     | 0.0391  | 0.0397 | 0.1809  | 0.2739 | 0.0205 | 0.0310 | 0.0438 | 0.0613 |
| sfmlearner (d) | 10    | 0.0372  | 0.0388 | 0.2996  | 0.3974 | 0.0194 | 0.0292 | 0.0427 | 0.0575 |
| sfmlearner (t) | 9     | 0.0347  | 0.0270 | 0.1508  | 0.1069 | 0.0147 | 0.0273 | 0.0410 | 0.0557 |
| sfmlearner (t) | 10    | 0.0262  | 0.0265 | 0.1434  | 0.1006 | 0.0119 | 0.0209 | 0.0300 | 0.0419 |



### Trajectory evaluation result

| model          | drive | interval | te_mean    | te_std     | re_mean   | re_std    |
|----------------|-------|----------|------------|------------|-----------|-----------|
| deepvofeat (d) | 09    | 1        | 50.880978  | 42.770940  | 5.863212  | 2.719989  |
| deepvofeat (d) | 09    | 2        | 50.992532  | 42.883168  | 5.867734  | 2.719449  |
| deepvofeat (d) | 09    | 3        | 50.967294  | 42.847035  | 5.867056  | 2.718938  |
| deepvofeat (d) | 09    | 4        | 51.083319  | 42.952792  | 5.874094  | 2.718504  |
| deepvofeat (d) | 10    | 1        | 58.754557  | 23.383633  | 5.157086  | 1.938336  |
| deepvofeat (d) | 10    | 2        | 58.817234  | 23.371211  | 5.164869  | 1.942680  |
| deepvofeat (d) | 10    | 3        | 58.838523  | 23.348437  | 5.167043  | 1.936930  |
| deepvofeat (d) | 10    | 4        | 58.899452  | 23.312014  | 5.179273  | 1.948856  |
| geonet (t)     | 09    | 1        | 317.110237 | 204.937289 | 60.725549 | 32.612120 |
| geonet (t)     | 09    | 2        | 301.306583 | 197.004111 | 57.239281 | 33.039670 |
| geonet (t)     | 09    | 3        | 285.753932 | 186.510603 | 55.023999 | 35.009085 |
| geonet (t)     | 09    | 4        | 282.085802 | 182.293137 | 54.133437 | 35.378437 |
| geonet (t)     | 10    | 1        | 190.252887 | 145.445193 | 29.750839 | 16.312634 |
| geonet (t)     | 10    | 2        | 169.343088 | 126.710314 | 24.331270 | 13.104396 |
| geonet (t)     | 10    | 3        | 151.993435 | 112.752251 | 21.173155 | 11.081827 |
| geonet (t)     | 10    | 4        | 137.140937 | 102.198353 | 19.458990 | 10.203547 |
| orb_full (t)   | 09    | 1        | 3.267358   | 8.537666   | 0.527214  | 0.292962  |
| orb_full (t)   | 10    | 1        | 3.580680   | 2.512910   | 0.780063  | 0.276202  |
| sfmlearner (d) | 09    | 1        | 294.531290 | 179.173584 | 36.630341 | 12.433388 |
| sfmlearner (d) | 09    | 2        | 187.172722 | 111.703713 | 23.098457 | 8.750662  |
| sfmlearner (d) | 09    | 3        | 172.110927 | 101.050503 | 22.063986 | 8.571738  |
| sfmlearner (d) | 09    | 4        | 133.698398 | 82.457178  | 16.786935 | 7.212463  |
| sfmlearner (d) | 10    | 1        | 300.612322 | 213.094592 | 33.113980 | 16.164048 |
| sfmlearner (d) | 10    | 2        | 154.415732 | 98.189274  | 23.784812 | 10.595047 |
| sfmlearner (d) | 10    | 3        | 193.596702 | 124.826562 | 26.207837 | 11.552613 |
| sfmlearner (d) | 10    | 4        | 175.382717 | 115.465386 | 24.192112 | 11.988613 |
| sfmlearner (t) | 09    | 1        | 67.989706  | 39.503483  | 9.049760  | 3.752482  |
| sfmlearner (t) | 09    | 2        | 60.607590  | 36.847762  | 8.475532  | 3.695645  |
| sfmlearner (t) | 09    | 3        | 57.565464  | 38.224664  | 8.336646  | 3.947951  |
| sfmlearner (t) | 09    | 4        | 53.113459  | 30.621533  | 7.060156  | 2.996655  |
| sfmlearner (t) | 10    | 1        | 80.365616  | 57.599062  | 8.666806  | 3.633308  |
| sfmlearner (t) | 10    | 2        | 59.299741  | 43.690864  | 6.532136  | 2.735147  |
| sfmlearner (t) | 10    | 3        | 63.476712  | 47.178590  | 6.581644  | 2.725968  |
| sfmlearner (t) | 10    | 4        | 40.068201  | 28.534463  | 4.598471  | 1.761948  |

