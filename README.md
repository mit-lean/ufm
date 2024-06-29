Uncertainty from Motion (UfM) for Monocular Depth Estimation
============================
Efficient uncertainty estimation requiring only one inference per input by merging predictions across multiple views in a video sequence. 

Lab website: [lean.mit.edu](https://lean.mit.edu/)
## Introduction
This repo contains code to run the algorithm Uncertainty from Motion (UfM) for DNN monocular depth estimation, published in our paper [[1]](#1). 

Abstract: Deployment of deep neural networks (DNNs) for monocular depth estimation in safety-critical scenarios on resource-constrained platforms requires well-calibrated and efficient uncertainty estimates. However, many popular uncertainty estimation techniques, including state-of-the-art ensembles and popular sampling-based methods, require multiple inferences per input, making them difficult to deploy in latency-constrained or energy-constrained scenarios. We propose a new algorithm, called Uncertainty from Motion (UfM), that requires only one inference per input. UfM exploits the temporal redundancy in video inputs by merging incrementally the per-pixel depth prediction and per-pixel aleatoric uncertainty prediction of points that are seen in multiple views in the video sequence. When Ensemble-UfM is applied to ensembles, we show that UfM can retain the uncertainty quality of ensembles at a fraction of the energy by running only a single ensemble member at each frame and fusing the uncertainty over the sequence of frames.

## File structure  
* [src](src) : main UfM code
* [examples](examples) : examples to run UfM or calibration 
* [libs](libs) : contains calibration submodule
* [configuration](configuration) : configuration yaml files 
* [dataloaders](dataloaders) : reading in dataset formats 
* [nets](nets) : DNN architecture for given trained ensemble 
* [pre_process](pre_process) : pre-processing for example
* [post_process](post_process) : visualizing results
## Dependencies
This code was tested with Python 3.9 and PyTorch 1.13.1 on Ubuntu 20.04 with Miniconda. To install the dependencies, create and activate a new conda environment with Python 3.9, install PyTorch from [source](https://pytorch.org/get-started/locally/), recursively clone this repo, and run `installation.sh` to install the rest of the dependencies into the conda environment. 
```bash
(conda_env) foo@bar:~$ git clone --recurse-submodules https://github.com/mit-lean/ufm.git uncertainty_from_motion
(conda_env) foo@bar:~$ cd uncertainty_from_motion
(conda_env) foo@bar:~/uncertainty_from_motion$ ./installation.sh
```
## Get started
In order to include Ensemble-UfM in an existing pipeline with your own trained ensemble, make a configuration YAML file for your sequence; examples are in the [configuration/examples](configuration/examples) folder in this repo. Then, place this repo (`uncertainty_from_motion`) under a `libs` folder within your project, and modify the following pseudo-code to call UfM from your project. 
```python
from libs.uncertainty_from_motion.src import UncertaintyFromMotion, run_UfM_on_frame
run_parameters = <READ IN CONFIGURATION YAML> # initialize parameters from configuration file
UfM_pc = UncertaintyFromMotion(run_parameters, torch.device(ufm_device),train_dataset_idx) # initialize point cloud
models = <INITIALIZE LIST OF ENSEMBLE MEMBERS> # read in list of ensemble member DNNs
for i, data in enumerate(train_loader): # enumerate through a train_loader where data contains input, target, translation, rotation
    model_num = i % len(models) # select model to run on this image
    depth, aleatoric_var = <RUN FORWARD INFERENCE WITH models[model_num] ON iTH INPUT> # run forward inference to obtain depth prediction and aleatoric variance prediction on this image
    trans, rot = <GET TRANSLATION AND ROTATION TENSORS FOR CAMERA AT THIS IMAGE> # get translation/rotation tensors for the camera pose at this image
    UfM_uncertainty = run_UfM_on_frame(depth, aleatoric_var, trans, rot, UfM_pc, i) # run UfM to get UfM combined uncertainty (epistemic and aleatoric)
    # UfM_epistemic_uncertainty = run_UfM_on_frame(depth, torch.zeros(depth.shape, device=torch.device(ufm_device)), trans, rot, UfM_pc, i) # run this instead to just get UfM epistemic uncertainty by passing in zero aleatoric uncertainty
```
## Running examples
We present an example for running UfM on the TUM dataset sequences [[2]](#2) using groundtruth poses and FCDenseNet [[3]](#3) trained on NYUDepthV2 [[4]](#4) dataset with a modified architecture and loss function that predicts both depth and aleatoric uncertainty as in [[5]](#5), as an ensemble (size = 10) randomly initialized and trained as in [[6]](#6). 
### Download dataset
Download a video sequence from the TUM RGBD dataset [[2]](#2) [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/download). 
### Download trained ensemble or include your own trained ensemble 
Download the new trained FCDenseNet ensemble [here](https://drive.google.com/drive/folders/1VW0fcpQpm9lj9Kj1MUbKyY3s-R5IIKaj?usp=sharing). Alternatively, you can use this example code with your own trained ensemble or sampling-based method and modify the appropriate parameters in the configuration YAML file.  
### Process and store the poses as tensors 
Download the TUM RGBD helper script `associate.py` from [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools) and place inside the [dataloaders](dataloaders) directory. Replace `<DATASET_SEQ_PATH>` with correct path and run 
```
(conda_env) foo@bar:~/uncertainty_from_motion$ pre_process/process_poses_tum_dataset.sh <DATASET_SEQ_PATH>
```
The rotation and translation matrices for each frame will be stored under `DATASET_SEQ_PATH/pose_per_frame` in PyTorch tensor, NumPy, and csv formats. 
### Using files from existing works
Download the DNN architecture code from this [link](https://github.com/bfortuner/pytorch_tiramisu/blob/master/models/tiramisu.py) and metrics code from this [link](https://github.com/dwofk/fast-depth/blob/master/metrics.py). Then, run the following commands that modifies the code with an aleatoric head for the architecture code and to also track loss for the metrics code. 
```bash
(conda_env) foo@bar:~/uncertainty_from_motion$ python3 pre_process/modify_files_with_diff.py <PATH_TO_TIRAMISU.PY> ./nets/FCDenseNet.py ./nets/FCDenseNet_diff.json
(conda_env) foo@bar:~/uncertainty_from_motion$ python3 pre_process/modify_files_with_diff.py <PATH_TO_METRICS.PY> ./nets/metrics.py ./nets/metrics_diff.json
```
### Change parameters for run
We provide configuration files for running on TUM RGBD sequences under [configuration/examples](configuration/examples). Change the paths for the dataset directory `dataset_seq_path`, output directory `output_path`, pose directory `pose_path`, parent directory for trained models `trained_models_path`, individual directory for a trained model in `trained_model_folder`, and checkpoint model you want to test in `best_model_name`. 

Change `method` to one of the following methods. The bolded methods are the intended use cases of UfM. We recommend comparing the baseline of `ensemble` baseline to the `ensemble_UfM` to compare performance. 

Baselines: aleatoric, ensemble, MCDropout

UfM accelerated ensemble or MC-Dropout (one DNN inference per input): **ensemble_UfM**, **MCDroput_UfM**

UfM ablation experiments: aleatoric_UfM, full_ensemble_UfM, full_MCDropout_UfM

Note: the ablation experiments test the effect of UfM with different configurations; these methods do not improve computational efficiency. 
The parameters in the configuration file can be changed to fit your use case. `max_cloud_size` controls the maximum point cloud size used in UfM; setting it larger will increase overhead of UfM. In order to run  methods that require MC-Dropout, dropout probability `p` must be non-zero in the configuration file. The parameter `max_frames_to_save` sets the number of image results to save for calibration and visualization; if larger than number of images in sequence, all results will be saved. 
### Run UfM
Make sure parent folder repo is on the Python path, e.g., 
```
(conda_env) foo@bar:~/uncertainty_from_motion$ export PYTHONPATH="${PYTHONPATH}:<PATH_TO_PARENT_FOLDER>"
```
Then, run UfM using the example code and your configuration file: 
```
(conda_env) foo@bar:~/uncertainty_from_motion$ python3 examples/evaluate_uncertainty.py PATH_TO_CONFIGURATION_FILE
```
Results will be stored in the `output_path` set in the configuration file. 
### Evaluate uncertainty quality 
To evaluate the uncertainty quality, we compute the expected calibration error for a $\delta_1$ vs. confidence plot as discussed in [[1]](#1). In order to run the calibration, make sure you have recursively downloaded the contents of the `uncertainty_calibration` submodule within `libs`; if `uncertainty_from_motion/libs/uncertainty_calibration/` is empty, run: `git submodule update --init --recursive` within submodule folder.
Then, use the calibration example we have provided to compute calibration curves and expected calibration error (ECE) on the saved results. 
```
(conda_env) foo@bar:~/uncertainty_from_motion$ python3 examples/calibrate.py <UNCERTAINTY_TYPE> <PATH_TO_SAVED_RESULTS> <PATH_TO_OUTPUT_DIRECTORY>
```
where `PATH_TO_SAVED_RESULTS` is the folder with the UfM per frame results given by `os.path.join(output_path, method,'per_frame_data')` where `output_path` and `method` are set in the configuration file, `PATH_TO_OUTPUT_DIRECTORY` is a user-defined path for where the calibration results and plots should be stored, and `UNC_TYPE` is `epistemic_plus_aleatoric` for baselines (ensemble, MCDropout, aleatoric) and `multiview` for proposed UfM and ablation methods (ensemble-UfM, MCDropout-UfM, aleatoric-UfM, full ensemble-UfM, full MCDropout-UfM). 

A sample of results (ECE ($\delta_1$ accuracy)) using the given trained FCDenseNet ensemble are given below. Ensemble-UfM can maintain ensemble uncertainty quality at a fraction of the computational cost. 
| Sequence | Aleatoric  (baseline) | Ensemble (baseline) | Ensemble-UfM (proposed method) | MCDropout  (baseline)  | MCDropout-UfM (proposed method) | Aleatoric-UfM  (ablation) | Full ensemble-UfM (ablation)  | Full MCDropout-UfM (ablation) |
| :---:                             | :---: | :---: | :---:   | :---: | :---: | :---:   | :---: | :---: |
| TUM-RGBD room                     | 0.42 (46.6%) | 0.34 (48.8%) | 0.30 (49.1%) | 0.39 (47.7%) | 0.34 (47.7%) | 0.36 (46.6%) | 0.29 (48.8%) | 0.33 (47.8%) |
| TUM-RGBD long_office_household    | 0.36 (55.9%) | 0.30 (56.9%) | 0.29 (56.2%) | 0.35 (55.2%) | 0.32 (55.0%) | 0.33 (55.9%) | 0.28 (56.9%) | 0.32 (55.2%) |
| TUM-RGBD desk                     | 0.59 (30.8%) | 0.50 (31.0%) | 0.46 (31.7%) | 0.57 (30.9%) | 0.52 (30.9%) | 0.52 (30.8%) | 0.45 (31.0%) | 0.51 (31.0%) |
| TUM-RGBD pioneer slam3            | 0.53 (37.5%) | 0.39 (43.4%) | 0.36 (40.0%) | 0.51 (38.1%) | 0.45 (37.8%) | 0.46 (37.5%) | 0.33 (43.4%) | 0.44 (38.0%) |
| ScanNet 0092_01                   | 0.34 (61.0%) | 0.31 (60.6%) | 0.29 (60.3%) | 0.30 (64.1%) | 0.28 (63.8%) | 0.31 (61.0%) | 0.28 (60.6%) | 0.27 (64.1%) |
| ScanNet 0101_01                   | 0.19 (77.5%) | 0.15 (77.9%) | 0.15 (75.3%) | 0.19 (77.2%) | 0.16 (76.7%) | 0.16 (77.5%) | 0.12 (77.9%) | 0.16 (77.3%) |

Note: MC-Dropout is stochastic so results may vary for those methods. In addition, results will vary from [[1]](#1) due to a correction and new ensemble with correct weight initialization. 
### Visualize results 
To store a video of the results, run 
```
(conda_env) foo@bar:~/uncertainty_from_motion$ python3 post_process/visualize_video_rgb_gt_pred_error_totalunc_nll.py <PATH_TO_RESULTS_DIR> <PATH_TO_OUTPUT_VIDEO_DIR> <INPUT_HEIGHT> <INPUT_WIDTH> <UNC_TYPE>
```
where `input_height` and `input_width` are the same values as in configuration file, and `UNC_TYPE` is `epistemic_plus_aleatoric` for baselines (ensemble, MCDropout, aleatoric) and `multiview` for proposed UfM and ablation methods (ensemble-UfM, MCDropout-UfM, aleatoric-UfM, full ensemble-UfM, full MCDropout-UfM). 

## Citation
If you reference this repo, consider citing the following:
```
@inproceedings{sudhakar2022uncertainty,
  title={Uncertainty from Motion for DNN Monocular Depth Estimation},
  author={Sudhakar, Soumya and Sze, Vivienne and Karaman, Sertac},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)},
  pages={8673--8679},
  year={2022},
  organization={IEEE}
}
```
## References
<a id="1">[1]</a> 
Sudhakar, Soumya, Vivienne Sze, and Sertac Karaman. "Uncertainty from Motion for DNN Monocular Depth Estimation." 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022.

<a id="2">[2]</a> 
Sturm, Jürgen, et al. "A Benchmark for the Evaluation of RGB-D SLAM Systems." 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2012.

<a id="3">[3]</a> 
Jégou, Simon, et al. "The One Hundred Layers Tiramisu: Fully Convolutional Densenets for Semantic Segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2017.

<a id="4">[4]</a> 
Silberman, Nathan, et al. "Indoor Segmentation and Support Inference from RGBD Images." European Conference on Computer Vision. Springer, Berlin, Heidelberg, 2012.

<a id="5">[5]</a> 
Kendall, Alex, and Yarin Gal. "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?." Advances in Neural Information Processing Systems 30 (2017).

<a id="6">[6]</a> 
Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." Advances in Neural Information Processing Systems 30 (2017).
