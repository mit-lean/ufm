import os
import time
import csv
import numpy as np
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

from uncertainty_from_motion.nets.metrics import AverageMeter, Result
from uncertainty_from_motion.nets import criteria
from uncertainty_from_motion.src import utils
from uncertainty_from_motion.dataloaders.tum_dataloader import TUMMetaDataset
import uncertainty_from_motion.dataloaders.scannet_dataloader as scannet_dataloader

import math
import pandas as pd
from pprint import pprint
from uncertainty_from_motion.src import UncertaintyFromMotion, run_UfM_on_frame
from PIL import Image # store as PNG

# initialize global variables and update path
run_parameters = utils.initiate_parameters(sys.argv[1])
global output_directory

def create_data_loaders():
    # Create data loader for validation (returns empty data loader for train loader, not training in this code)
    print("=> creating data loaders ...")
    train_loader = None # returning empty training dataset, UfM used during validation only
    val_loader = None # initialize validation laoder 
    print("Validation directory: {}".format(run_parameters['dataset_seq_path']))
    # set up validation video sequences based on run_parameters; currently supports datasets in TUM RGBD and ScanNet format
    if run_parameters['dataset'] == 'tum':
        val_dataset = TUMMetaDataset(run_parameters['dataset_seq_path'])
    elif run_parameters['dataset'] == 'scannet':
        val_dataset = scannet_dataloader.ScanNetMetaDataset(root=run_parameters['dataset_seq_path'])
    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=run_parameters['workers'], pin_memory=True)
    print("=> data loaders created.")
    return train_loader, val_loader

def save_files(output_uncertainty_dir, gt_np_array, rgb_np_array, pred_np_array, aleatoric_np_array, epistemic_np_array, multiview_np_array, error_np_array, NLL_np_array, rgb_img_np_array):
    # save groundtruth, prediction, rgb (np array, image), aleatoric uncertainty, epistemic uncertainty, multiview uncertainty, error, NLL as NumPy arrays
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/gt.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/gt.npy",gt_np_array)
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/rgb_input.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/rgb_input.npy",rgb_np_array)
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/pred.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/pred.npy",pred_np_array)
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/aleatoric_uncertainty.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/aleatoric_uncertainty.npy",aleatoric_np_array)
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/epistemic_uncertainty.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/epistemic_uncertainty.npy",epistemic_np_array)
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/multiview_uncertainty.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/multiview_uncertainty.npy",multiview_np_array)
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/error.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/error.npy",error_np_array)
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/NLL.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/NLL.npy",error_np_array)
    rgb_img = Image.fromarray(rgb_img_np_array.astype('uint8'))
    os.makedirs(os.path.dirname(output_uncertainty_dir), exist_ok=True)
    rgb_img.save(output_uncertainty_dir + "/rgb_input.png")

def print_avg_stats(avg):
    # print average stats of validation set 
    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={average.gpu_time:.3f}\n'.format(
        average=avg))

def print_iteration_stats(i, total_val, gpu_time, result, average_meter):
    # print statistics during validation iterations
    print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Loss={result.loss:.3f}({average.loss:.3f}) '.format(
                   i+1, total_val, gpu_time=gpu_time, result=result, average=average_meter.average()))
    
def write_stats_to_file(test_csv, avg, avg_NLL, network_times, method_times, multiview_times):
    # write out average statistics and timing data to a text file
    fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time', 'loss', 'avg_NLL', 
                'network_time', 'method_time', 'multiview_time']
    with open(test_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
        'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
        'data_time': avg.data_time, 'gpu_time': avg.gpu_time, 'loss': avg.loss, 'avg_NLL': avg_NLL, 
        'network_time': np.average(network_times), 'method_time': np.average(method_times), 'multiview_time': np.average(multiview_times)})
    return

def run_forward_inference(run_parameters, model, input):
    if run_parameters['arch'] == 'FCDenseNet':
        output = model(input)
    else:
        print(run_parameters['arch'] + " architecture not suported yet.")
    return output

def run_method():
    # selects which method(s) we run 
    # initialize and print global variables
    global output_directory, best_result, run_parameters
    print(run_parameters)
    # select loss function (criterion); used just to compute loss on validation set, not training
    if run_parameters['loss'] == "heteroscedastic":
        criterion = criteria.HeteroscedasticLoss().cuda()
    else:
        print("This is not a valid loss function. Valid value of run_parameters['loss'] is heteroscedastic.")
    # during inference time 
    if run_parameters['evaluate']:
        models = [] # initialize list of models
        cores_by_ensemble_mem = core_manager() # initialize list of cores
        # make output directory for results to be stored
        output_directory = os.path.join(run_parameters['output_path']) 
        print("Output directory: {}".format(output_directory))
        if not os.path.exists(output_directory): # create results folder, if it does not already exist
            os.makedirs(output_directory)
        # pretty print run parameters to text file
        with open(os.path.join(output_directory,'run_parameters_validate.txt'), 'wt') as out:
            pprint(run_parameters, stream=out)
        if 'arch' not in run_parameters:
            run_parameters['arch'] = 'FCDenseNet' # backward compatibility
        if run_parameters['arch'] == 'FCDenseNet':
            # load in ensemble (for non-ensemble methods, ensemble size = 1) on to correct GPU
            checkpoints = [] # initialize empty container
            for ensemble_member_number in range(0, run_parameters['ensemble_size']):
                assert os.path.isfile(utils.get_trained_model_directory(run_parameters,ensemble_member_number+1)+ "/" + run_parameters['best_model_name']),\
                    "=> no best model found at '{}'".format(utils.get_trained_model_directory(run_parameters,ensemble_member_number+1)+"/" + run_parameters['best_model_name'])
                print("=> loading best model '{}'".format(utils.get_trained_model_directory(run_parameters,ensemble_member_number+1)+"/" + run_parameters['best_model_name']))
                if run_parameters['device'] == 'GPU':
                    cuda_core = ensemble_member_number % run_parameters['num_cores']
                    print("cuda core: " + str(cuda_core))
                    checkpoints.append(torch.load(utils.get_trained_model_directory(run_parameters,ensemble_member_number+1)+"/" + run_parameters['best_model_name'], map_location=torch.device('cuda:'+str(cuda_core))))
                else: # CPU
                    checkpoints.append(torch.load(utils.get_trained_model_directory(run_parameters,ensemble_member_number+1)+"/" + run_parameters['best_model_name'], map_location=torch.device('cpu')))
            # add all models to list of models (e.g., ensemble size number of models for ensemble, 1 model for MC-Dropout) 
            for m in range(0, run_parameters['ensemble_size']):
                models.append(checkpoints[m]['model'])
        else: # add here for new architectures to support 
            print(run_parameters['arch'] + " architecture not suported yet.")
        _, val_loader = create_data_loaders() # create data loaders 
        # select method based on run_parameters and run/store results 
        if float(run_parameters['p'])==0: # non MC-Dropout variants
            if "aleatoric" in run_parameters['method']: # baseline
                validate_aleatoric(run_parameters, val_loader, models, criterion, True) # evaluate on validation set
            if "ensemble" in run_parameters['method']: # baseline
                validate_ensemble(run_parameters, val_loader, models, criterion, True)
            if "ensemble_UfM" in run_parameters['method']: # Ensemble-UfM
                validate_ensemble_UfM(run_parameters, val_loader, models, criterion, True)
            if "aleatoric_UfM" in run_parameters['method']: # ablation study 
                validate_aleatoric_UfM(run_parameters, val_loader, models, criterion, True)
            if "full_ensemble_UfM" in run_parameters['method']: # ablation study 
                validate_full_ensemble_UfM(run_parameters, val_loader, models, criterion, True)
        else: # MC-Dropout variants (p non-zero)
            if "MCDropout" in run_parameters['method']: # baseline
                validate_MCDropout(run_parameters, val_loader, models, criterion, True) 
            if "MCDropout_UfM" in run_parameters['method']: # MCDropout-UfM
                validate_MCDropout_UfM(run_parameters, val_loader, models, criterion, True)
            if "full_MCDropout_UfM" in run_parameters['method']: # ablation study 
                validate_full_MCDropout_UfM(run_parameters, val_loader, models, criterion, True) 
        return
    else:
        print("This code is for evaluation only, not training. Change run_paramters['evaluate'] to run.")
    
def validate_aleatoric(run_parameters, val_loader, models, criterion, write_to_file=True):
    print("Validating aleatoric:")
    global output_directory
    average_meter = AverageMeter() # initialize avg result metric
    # set up model in evaluation mode
    model = models[0] # select first model to test 
    if run_parameters['device'] == 'CPU':
        model.cpu() # switch to CPU
    model.eval() # switch to evaluation mode
    # intialize time and avg metrics
    network_times = []
    method_times = []
    avg_NLL_sum = 0
    # run method for all frames in validation video sequence 
    for i, metadata in enumerate(val_loader):
        # load in data 
        input = metadata['rgb']
        target = metadata['depth']
        tic_data = time.time() # used for data_time timing
        # move input/target to correct correct GPU
        input = input.cuda(0)
        target = target.cuda(0)
        torch.cuda.synchronize(device=torch.device("cuda:0"))
        data_time = time.perf_counter() - tic_data
        tic_gpu = time.perf_counter()
        tic_method = time.perf_counter()
        # run DNN on this frame 
        with torch.no_grad():
            tic = time.perf_counter()
            output = run_forward_inference(run_parameters, model, input) # model(input)
            torch.cuda.synchronize(device="cuda:0")
            toc = time.perf_counter()
            if i > 40: # warm start for timing
                network_times.append(toc-tic)
            # parse depth prediction, aleatoric uncertainty, epistemic uncertainty (if present) from DNN output
            aleatoric_unc = parse_aleatoric_uncertainty(output)
            pred = parse_prediction(output)
            epistemic_unc = torch.zeros(output[:,0,:,:].shape, device=torch.device('cuda:0'))
        torch.cuda.synchronize(device="cuda:0")
        toc_method = time.perf_counter()
        if i > 40: # warm start for timing 
            method_times.append(toc_method-tic_method)
        gpu_time = time.perf_counter() - tic_gpu
        # measure accuracy and record loss
        result = Result()
        loss = criterion(output, target) # just for logging into result
        result.evaluate(pred.data, target.data, loss)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        # compute NLL 
        NLL = compute_NLL(pred, aleatoric_unc, target)
        avg_NLL_sum = avg_NLL_sum + torch.mean(NLL)
        # save files
        if ((i < run_parameters['max_frames_to_save']) or ((i % run_parameters['store_freq'] == 0) and run_parameters['max_frames_to_save'] != 0)): 
            output_uncertainty_dir = output_directory+"/aleatoric/per_frame_data/"+str(i)
            save_files(output_uncertainty_dir, gt_np_array=target.data.cpu().data.numpy(), rgb_np_array=np.squeeze(input.data.cpu().data.numpy())[0:3,:,:], pred_np_array=np.squeeze(pred.data.cpu().data.numpy()), aleatoric_np_array=np.squeeze(aleatoric_unc.data.cpu().data.numpy()), epistemic_np_array=np.squeeze(epistemic_unc.data.cpu().data.numpy()), multiview_np_array=np.zeros((224,224)), error_np_array=np.squeeze((target-pred).abs().data.cpu().data.numpy()), NLL_np_array=np.squeeze(NLL.data.cpu().data.numpy()), rgb_img_np_array=np.squeeze(input.data.cpu().data.numpy()).transpose(1,2,0)*255)
        if (i+1) % run_parameters['print_freq'] == 0:
            print_iteration_stats(i, total_val=len(val_loader), gpu_time=gpu_time, result=result, average_meter=average_meter)
    avg = average_meter.average()
    print_avg_stats(avg)
    avg_NLL = avg_NLL_sum.item()/float(len(val_loader))
    if write_to_file:
        os.makedirs(os.path.dirname(os.path.join(output_directory,"aleatoric")), exist_ok=True)
        test_csv = os.path.join(output_directory,"aleatoric", "test.csv")
        write_stats_to_file(test_csv=test_csv, avg=avg, avg_NLL=avg_NLL, network_times=network_times, method_times=method_times, multiview_times=np.zeros((1))) 
    return

def validate_ensemble(run_parameters, val_loader, models, criterion, write_to_file=True):
    print("Validating ensemble:")
    global output_directory
    average_meter = AverageMeter() # initialize avg result metric
    # set up model in evaluation mode
    if run_parameters['device'] == 'CPU':
        for m in range(0, run_parameters['ensemble_size']):
            models[m] = models[m].cpu() # switch to CPU
    for m in range(0, run_parameters['ensemble_size']):
        models[m] = models[m].eval() # switch to evaluation mode
    # intialize time and avg metrics
    network_times = []
    method_times = [] 
    avg_NLL_sum = 0
    # run method for all frames in validation video sequence
    for i, metadata in enumerate(val_loader):
        # load in data 
        input = metadata['rgb']
        target = metadata['depth']
        # initialize partial mean and variance sums with zero arrays
        partial_mean = torch.zeros((run_parameters['input_height'],run_parameters['input_width']), device = torch.device('cuda:0'))
        partial_aleatoric_unc = torch.zeros((run_parameters['input_height'],run_parameters['input_width']), device = torch.device('cuda:0'))
        partial_epistemic_unc = torch.zeros((run_parameters['input_height'],run_parameters['input_width']), device = torch.device('cuda:0'))
        data_time = 0 # initialize timer sum 
        gpu_time = 0 # initialize timer sum
        network_time_sum = 0
        tic_method = time.perf_counter()
        # run DNN ensemble on this frame 
        for ensemble_member_number in range(0, run_parameters['ensemble_size']):
            tic_data = time.perf_counter()
            cuda_core = ensemble_member_number % run_parameters['num_cores']
            input = input.cuda(cuda_core)
            target = target.cuda(cuda_core)
            torch.cuda.synchronize(device=torch.device("cuda:"+str(cuda_core)))
            data_time += (time.perf_counter() - tic_data)
            gpu_start = time.perf_counter()
            with torch.no_grad():
                tic_network = time.perf_counter()
                output = run_forward_inference(run_parameters, models[ensemble_member_number], input) # models[ensemble_member_number](input)
                torch.cuda.synchronize(device="cuda:"+str(cuda_core))
                toc_network = time.perf_counter()
                network_time_sum += (toc_network-tic_network)
                # parse depth prediction, aleatoric uncertainty, epistemic uncertainty (if present) from DNN output
                aleatoric_unc_single = parse_aleatoric_uncertainty(output)
                pred = parse_prediction(output)
                # calculate iterative mean, iterative variance for each of the subdirectories
                partial_mean = partial_mean.to(cuda_core)
                partial_mean = partial_mean + pred
                partial_aleatoric_unc = partial_aleatoric_unc.to(cuda_core)
                partial_aleatoric_unc = partial_aleatoric_unc + aleatoric_unc_single
                partial_epistemic_unc = partial_epistemic_unc.to(cuda_core)
                partial_epistemic_unc = partial_epistemic_unc + torch.pow(pred,2)
            torch.cuda.synchronize(device="cuda:"+str(cuda_core))
            gpu_time += (time.perf_counter() - gpu_start)
        # after running full ensemble on frame, divide by ensemble size to calculate variance
        ensemble_mean = (partial_mean /run_parameters['ensemble_size']).to('cuda:0')
        aleatoric_unc = (partial_aleatoric_unc/run_parameters['ensemble_size']).to('cuda:0')
        epistemic_unc = partial_epistemic_unc.to('cuda:0')/run_parameters['ensemble_size']-torch.pow(ensemble_mean.to('cuda:0'),2)
        ensemble_unc = aleatoric_unc + epistemic_unc 
        toc_method = time.perf_counter() 
        if i > 40: # warm start
            method_times.append((toc_method-tic_method)-data_time)
        if i > 40: # warm start
            network_times.append(network_time_sum)
        # measure accuracy and record loss
        result = Result()
        target = target.to('cuda:0')
        output = output.to('cuda:0')
        loss = criterion(output, target) # just for logging into result # TODO: output or ensemble_mean? 
        result.evaluate(ensemble_mean.data, target.data, loss)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        # compute NLL
        NLL = compute_NLL(ensemble_mean, ensemble_unc, target)
        avg_NLL_sum = avg_NLL_sum + torch.mean(NLL)
        # save files
        if ((i < run_parameters['max_frames_to_save']) or ((i % run_parameters['store_freq'] == 0) and run_parameters['max_frames_to_save'] != 0)): 
            output_uncertainty_dir = output_directory+"/"+"/ensemble/per_frame_data/"+str(i)
            save_files(output_uncertainty_dir, gt_np_array=target.data.cpu().data.numpy(), rgb_np_array=np.squeeze(input.data.cpu().data.numpy())[0:3,:,:], pred_np_array=np.squeeze(ensemble_mean.data.cpu().data.numpy()), aleatoric_np_array=np.squeeze(aleatoric_unc.data.cpu().data.numpy()), epistemic_np_array=np.squeeze(epistemic_unc.data.cpu().data.numpy()), multiview_np_array=np.zeros((224,224)), error_np_array=np.squeeze((target-ensemble_mean).abs().data.cpu().data.numpy()), NLL_np_array=np.squeeze(NLL.data.cpu().data.numpy()), rgb_img_np_array=np.squeeze(input.data.cpu().data.numpy()).transpose(1,2,0)*255)
        if (i+1) % run_parameters['print_freq'] == 0:
            print_iteration_stats(i, total_val=len(val_loader), gpu_time=gpu_time, result=result, average_meter=average_meter)
    avg = average_meter.average()
    print_avg_stats(avg)
    avg_NLL = avg_NLL_sum.item()/float(len(val_loader))
    if write_to_file:
        os.makedirs(os.path.dirname(os.path.join(output_directory,"ensemble")), exist_ok=True)
        test_csv = os.path.join(output_directory,"ensemble", "test.csv")
        write_stats_to_file(test_csv=test_csv, avg=avg, avg_NLL=avg_NLL, network_times=network_times, method_times=method_times, multiview_times=np.zeros((1)))
    return

def validate_aleatoric_UfM(run_parameters, val_loader, models, criterion, write_to_file=True):
    print("Validating aleatoric-UfM:")
    global output_directory
    average_meter = AverageMeter() # initialize avg result metric
    # set up model in evaluation mode
    model = models[0] # select first model to test 
    if run_parameters['device'] == 'CPU':
        model.cpu() # switch to CPU
    model.eval() # switch to evaluation mode
    # intialize time and avg metrics
    network_times = []
    method_times = []
    avg_NLL_sum = 0
    # initialize UfM parameters
    multiview_times = []
    PC = UncertaintyFromMotion(run_parameters, torch.device('cuda:0'))  # initialize point cloud
    # run method for all frames in validation video sequence
    for i, metadata in enumerate(val_loader):
        # load in data 
        input = metadata['rgb']
        target = metadata['depth']
        tic_data = time.perf_counter()
        # move input/target to correct correct GPU
        input = input.cuda(0)
        target = target.cuda(0)
        torch.cuda.synchronize(device=torch.device("cuda:0"))
        data_time = time.time() - tic_data
        tic_gpu = time.perf_counter()
        tic_method = time.perf_counter()
        # run DNN on this frame
        with torch.no_grad():
            tic_network = time.perf_counter()
            output = run_forward_inference(run_parameters, model, input) # model(input)
            torch.cuda.synchronize(device="cuda:0")
            toc_network = time.perf_counter()
            if i > 40: # warm start
                network_times.append(toc_network-tic_network)
            # parse depth prediction, aleatoric uncertainty, epistemic uncertainty (if present) from DNN output
            aleatoric_unc = parse_aleatoric_uncertainty(output)
            pred = parse_prediction(output)
            epistemic_unc = torch.zeros(output[:,0,:,:].shape, device=torch.device('cuda:0'))
        torch.cuda.synchronize(device="cuda:0")
        toc_method = time.perf_counter()
        if i > 40: # warm start
            method_times.append(toc_method-tic_method)
        gpu_time = time.perf_counter() - tic_gpu
        ## begin UfM
        # load rotation and translation
        translation = torch.load(run_parameters['pose_path'] +"/"+str(i)+"/translation.pt", map_location=torch.device('cuda:0'))
        rotation = torch.load(run_parameters['pose_path'] +"/"+str(i)+"/rotation.pt", map_location=torch.device('cuda:0'))
        # run multiview uncertainty
        tic_multiview = time.perf_counter()
        multiview_uncertainty = run_UfM_on_frame(pred, aleatoric_unc, translation, rotation, PC, i)
        torch.cuda.synchronize(device="cuda:0")
        toc_multiview = time.perf_counter()
        if i > 40: # warm start
            multiview_times.append(toc_multiview-tic_multiview)
        ## end UfM
        # measure accuracy and record loss
        result = Result()
        loss = criterion(output, target) # just for logging into result
        result.evaluate(pred.data, target.data, loss)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        # compute NLL
        NLL = compute_NLL(pred, multiview_uncertainty, target)
        avg_NLL_sum = avg_NLL_sum + torch.mean(NLL)
        # save files
        if ((i < run_parameters['max_frames_to_save']) or ((i % run_parameters['store_freq'] == 0) and run_parameters['max_frames_to_save'] != 0)):
            output_uncertainty_dir = output_directory+"/aleatoric_UfM/per_frame_data/"+str(i)
            save_files(output_uncertainty_dir, gt_np_array=target.data.cpu().data.numpy(), rgb_np_array=np.squeeze(input.data.cpu().data.numpy())[0:3,:,:], pred_np_array=np.squeeze(pred.data.cpu().data.numpy()), aleatoric_np_array=np.squeeze(aleatoric_unc.data.cpu().data.numpy()), epistemic_np_array=np.squeeze(epistemic_unc.data.cpu().data.numpy()), multiview_np_array=np.squeeze(multiview_uncertainty.data.cpu().data.numpy()), error_np_array=np.squeeze((target-pred).abs().data.cpu().data.numpy()), NLL_np_array=np.squeeze(NLL.data.cpu().data.numpy()), rgb_img_np_array=np.squeeze(input.data.cpu().data.numpy()).transpose(1,2,0)*255)
        if (i+1) % run_parameters['print_freq'] == 0:
            print_iteration_stats(i, total_val=len(val_loader), gpu_time=gpu_time, result=result, average_meter=average_meter)
    avg = average_meter.average()
    print_avg_stats(avg)
    print("Average UfM latency: " + str(np.average(multiview_times)))
    avg_NLL = avg_NLL_sum.item()/float(len(val_loader))
    if write_to_file:
        os.makedirs(os.path.dirname(os.path.join(output_directory,"aleatoric_UfM")), exist_ok=True)
        test_csv = os.path.join(output_directory,"aleatoric_UfM", "test.csv")
        write_stats_to_file(test_csv=test_csv, avg=avg, avg_NLL=avg_NLL, network_times=network_times, method_times=method_times, multiview_times=multiview_times)
    return

def validate_full_ensemble_UfM(run_parameters, val_loader, models, criterion, write_to_file=True):
    print("Validating full ensemble-UfM:")
    global output_directory
    average_meter = AverageMeter() # initialize avg result metric
    # set up model in evaluation mode
    if run_parameters['device'] == 'CPU':
        for m in range(0, run_parameters['ensemble_size']):
            models[m] = models[m].cpu() # switch to CPU
    for m in range(0, run_parameters['ensemble_size']):
        models[m] = models[m].eval() # switch to evaluation mode
    # intialize time and avg metrics
    network_times = []
    method_times = [] 
    avg_NLL_sum = 0
    # initialize UfM parameters
    multiview_times = []
    PC = UncertaintyFromMotion(run_parameters, torch.device('cuda:0'))  # initialize point cloud
    # run method for all frames in validation video sequence
    for i, metadata in enumerate(val_loader):
        # load in data 
        input = metadata['rgb']
        target = metadata['depth']
        # initialize partial mean and variance sums with zero arrays
        partial_mean = torch.zeros((run_parameters['input_height'],run_parameters['input_width']), device = torch.device('cuda:0'))
        # partial_variance = torch.zeros((run_parameters['input_height'],run_parameters['input_width']), device = torch.device('cuda:0'))
        partial_aleatoric_unc = torch.zeros((run_parameters['input_height'],run_parameters['input_width']), device = torch.device('cuda:0'))
        partial_epistemic_unc = torch.zeros((run_parameters['input_height'],run_parameters['input_width']), device = torch.device('cuda:0'))
        # initialize timing
        data_time = 0  
        gpu_time = 0 
        network_times_sum = 0
        tic_method = time.perf_counter()
        # run DNN on this frame using selected GPU based on number of GPUs
        for ensemble_member_number in range(0, run_parameters['ensemble_size']):
            tic_data = time.perf_counter()
            cuda_core = ensemble_member_number % run_parameters['num_cores']
            input = input.cuda(cuda_core)
            target = target.cuda(cuda_core)
            torch.cuda.synchronize(device=torch.device("cuda:"+str(cuda_core)))
            data_time += (time.perf_counter() - tic_data)
            gpu_start = time.perf_counter()
            with torch.no_grad():
                tic_network = time.perf_counter()
                output = run_forward_inference(run_parameters, models[ensemble_member_number], input) # models[ensemble_member_number](input)
                torch.cuda.synchronize(device="cuda:"+str(cuda_core))
                toc_network = time.perf_counter() 
                network_times_sum += (toc_network-tic_network)
                # parse depth prediction, aleatoric uncertainty, epistemic uncertainty (if present) from DNN output
                aleatoric_unc_single = parse_aleatoric_uncertainty(output)
                pred = parse_prediction(output)
                # calculate iterative mean, iterative variance for each of the subdirectories
                partial_mean = partial_mean.to(cuda_core)
                partial_mean = partial_mean + pred
                partial_aleatoric_unc = partial_aleatoric_unc.to(cuda_core)
                partial_aleatoric_unc = partial_aleatoric_unc + aleatoric_unc_single
                partial_epistemic_unc = partial_epistemic_unc.to(cuda_core)
                partial_epistemic_unc = partial_epistemic_unc + torch.pow(pred,2)
            torch.cuda.synchronize(device="cuda:"+str(cuda_core))
            gpu_time += (time.perf_counter() - gpu_start)
        # if last time, divide by n to store variance, and store all values
        ensemble_mean = (partial_mean /run_parameters['ensemble_size']).to('cuda:0')
        aleatoric_unc = (partial_aleatoric_unc/run_parameters['ensemble_size']).to('cuda:0')
        epistemic_unc = partial_epistemic_unc.to('cuda:0')/run_parameters['ensemble_size']-torch.pow(ensemble_mean.to('cuda:0'),2)
        ensemble_unc = aleatoric_unc + epistemic_unc 
        # print("torch.allclose((partial_variance/run_parameters['ensemble_size']-torch.pow(ensemble_mean,2)), (aleatoric_unc+epistemic_unc)): " + str(torch.allclose(epistemic_unc, (aleatoric_unc+epistemic_unc),rtol=1e-05, atol=1e-06)))
        toc_method = time.perf_counter()
        if i > 40: # warm start
            method_times.append((toc_method-tic_method)-data_time)
        # print(f"Inference on frame took {toc - tic:0.4f} seconds")
        if i > 40: # warm start
            network_times.append(network_times_sum)
        ## begin UfM
        # load rotation and translation
        translation = torch.load(run_parameters['pose_path'] +"/"+str(i)+"/translation.pt", map_location=torch.device('cuda:0'))
        rotation = torch.load(run_parameters['pose_path'] +"/"+str(i)+"/rotation.pt", map_location=torch.device('cuda:0'))
        # run multiview uncertainty
        tic_multiview = time.perf_counter()
        multiview_uncertainty = run_UfM_on_frame(ensemble_mean, ensemble_unc, translation, rotation, PC, i)
        torch.cuda.synchronize(device="cuda:0")
        toc_multiview = time.perf_counter()
        # print(f"Multiview on frame took {toc_multiview - tic_multiview:0.4f} seconds")
        if i > 40: # warm start
            multiview_times.append(toc_multiview-tic_multiview)
        ## end UfM
        # measure accuracy and record loss
        result = Result()
        output = output.to('cuda:0')
        target = target.to('cuda:0')
        loss = criterion(output, target) # just for logging into result
        result.evaluate(ensemble_mean.data, target.data, loss)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        # compute NLL
        NLL = compute_NLL(ensemble_mean, multiview_uncertainty, target)
        avg_NLL_sum = avg_NLL_sum + torch.mean(NLL)
        # save files
        if ((i < run_parameters['max_frames_to_save']) or ((i % run_parameters['store_freq'] == 0) and run_parameters['max_frames_to_save'] != 0)):
            output_uncertainty_dir = output_directory+"/full_ensemble_UfM/per_frame_data/"+str(i)
            save_files(output_uncertainty_dir, gt_np_array=target.data.cpu().data.numpy(), rgb_np_array=np.squeeze(input.data.cpu().data.numpy())[0:3,:,:], pred_np_array=np.squeeze(ensemble_mean.data.cpu().data.numpy()), aleatoric_np_array=np.squeeze(aleatoric_unc.data.cpu().data.numpy()), epistemic_np_array=np.squeeze(epistemic_unc.data.cpu().data.numpy()), multiview_np_array=np.squeeze(multiview_uncertainty.data.cpu().data.numpy()), error_np_array=np.squeeze((target-ensemble_mean).abs().data.cpu().data.numpy()), NLL_np_array=np.squeeze(NLL.data.cpu().data.numpy()), rgb_img_np_array=np.squeeze(input.data.cpu().data.numpy()).transpose(1,2,0)*255)
        if (i+1) % run_parameters['print_freq'] == 0:
            print_iteration_stats(i, total_val=len(val_loader), gpu_time=gpu_time, result=result, average_meter=average_meter)
    avg = average_meter.average()
    print_avg_stats(avg)
    avg_NLL = avg_NLL_sum.item()/float(len(val_loader))
    print("Average UfM latency: " + str(np.average(multiview_times)))
    if write_to_file:
        os.makedirs(os.path.dirname(os.path.join(output_directory,"full_ensemble_UfM")), exist_ok=True)
        test_csv = os.path.join(output_directory,"full_ensemble_UfM", "test.csv")
        write_stats_to_file(test_csv=test_csv, avg=avg, avg_NLL=avg_NLL, network_times=network_times, method_times=method_times, multiview_times=multiview_times)
    return

def core_manager():
    cores_by_ensemble_mem = []
    if (run_parameters['ensemble_size'] < run_parameters['num_cores']) or (run_parameters['ensemble_size'] == run_parameters['num_cores']):
        cores_by_ensemble_mem.extend(range(0, run_parameters['ensemble_size']))
    else: 
        cores_by_ensemble_mem.extend(range(0, run_parameters['num_cores']))
        idx_to_add = 0
        while (len(cores_by_ensemble_mem) < run_parameters['ensemble_size']):
            cores_by_ensemble_mem.append(cores_by_ensemble_mem[idx_to_add])
            idx_to_add = idx_to_add + 1
    return cores_by_ensemble_mem

def validate_ensemble_UfM(run_parameters, val_loader, models, criterion, write_to_file=True):
    print("Validating ensemble-UfM:")
    global output_directory
    average_meter = AverageMeter() # initialize avg result metric
    # set up model in evaluation mode
    if run_parameters['device'] == 'CPU':
        for m in range(0, run_parameters['ensemble_size']):
            models[m] = models[m].cpu() # switch to CPU
    for m in range(0, run_parameters['ensemble_size']):
        models[m] = models[m].eval() # switch to evaluation mode
    # intialize time and avg metrics
    network_times = []
    method_times = []
    avg_NLL_sum = 0
    # initialize UfM parameters
    multiview_times = []
    PC = UncertaintyFromMotion(run_parameters, torch.device('cuda:0'))  # initialize point cloud
    # initialize assignment of GPU to ensemble member
    cores_by_ensemble_mem = core_manager()
    # run method for all frames in validation video sequence
    for i, metadata in enumerate(val_loader):
        # load in data 
        input = metadata['rgb']
        target = metadata['depth']
        ensemble_member_number = i % run_parameters['ensemble_size'] 
        cuda_core = cores_by_ensemble_mem[ensemble_member_number]
        # move input/target to correct correct GPU
        tic_data = time.perf_counter()
        input = input.cuda(cuda_core)
        target = target.cuda(cuda_core)
        torch.cuda.synchronize(device=torch.device("cuda:"+str(cuda_core)))
        data_time = time.perf_counter() - tic_data
        # run DNN on this frame
        tic_gpu = time.perf_counter()
        tic_method = time.perf_counter()
        with torch.no_grad():
            tic_network = time.perf_counter()
            output = run_forward_inference(run_parameters, models[ensemble_member_number], input) # models[ensemble_member_number](input)
            torch.cuda.synchronize(device="cuda:"+str(cuda_core))
            toc_network = time.perf_counter()
            if i > 40: # warm start
                network_times.append(toc_network-tic_network)
            # parse depth prediction, aleatoric uncertainty, epistemic uncertainty (if present) from DNN output
            aleatoric_unc = parse_aleatoric_uncertainty(output)
            pred = parse_prediction(output)
            epistemic_unc = torch.zeros(output[:,0,:,:].shape, device=torch.device('cuda:0'))
        torch.cuda.synchronize(device="cuda:"+str(cuda_core))
        toc_method = time.perf_counter()
        if i > 40: # warm start
            method_times.append(toc_method-tic_method)
        gpu_time = time.perf_counter() - tic_gpu
        ## begin UfM
        # load rotation and translation
        translation = torch.load(run_parameters['pose_path'] +"/"+str(i)+"/translation.pt", map_location=torch.device('cuda:0'))
        rotation = torch.load(run_parameters['pose_path'] +"/"+str(i)+"/rotation.pt", map_location=torch.device('cuda:0'))
        # run multiview uncertainty
        tic_multiview = time.perf_counter()
        multiview_uncertainty = run_UfM_on_frame(pred.to('cuda:0'), aleatoric_unc.to('cuda:0'), translation, rotation, PC, i)
        torch.cuda.synchronize(device="cuda:0")
        toc_multiview = time.perf_counter()
        if i > 40: # warm start
            multiview_times.append(toc_multiview-tic_multiview)
        ## end UfM
        # measure accuracy and record loss
        result = Result()
        loss = criterion(output, target) # just for logging into result
        result.evaluate(pred.data, target.data, loss)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        # compute NLL
        NLL = compute_NLL(pred, multiview_uncertainty, target)
        avg_NLL_sum = avg_NLL_sum + torch.mean(NLL)
        # save files
        if ((i < run_parameters['max_frames_to_save']) or ((i % run_parameters['store_freq'] == 0) and run_parameters['max_frames_to_save'] != 0)):
            output_uncertainty_dir = output_directory+"/ensemble_UfM/per_frame_data/"+str(i)
            save_files(output_uncertainty_dir, gt_np_array=target.data.cpu().data.numpy(), rgb_np_array=np.squeeze(input.data.cpu().data.numpy())[0:3,:,:], pred_np_array=np.squeeze(pred.data.cpu().data.numpy()), aleatoric_np_array=np.squeeze(aleatoric_unc.data.cpu().data.numpy()), epistemic_np_array=np.squeeze(epistemic_unc.data.cpu().data.numpy()), multiview_np_array=np.squeeze(multiview_uncertainty.data.cpu().data.numpy()), error_np_array=np.squeeze((target-pred).abs().data.cpu().data.numpy()), NLL_np_array=np.squeeze(NLL.data.cpu().data.numpy()), rgb_img_np_array=np.squeeze(input.data.cpu().data.numpy()).transpose(1,2,0)*255)
        if (i+1) % run_parameters['print_freq'] == 0:
            print_iteration_stats(i, total_val=len(val_loader), gpu_time=gpu_time, result=result, average_meter=average_meter)
    avg = average_meter.average()
    print_avg_stats(avg)
    avg_NLL = avg_NLL_sum.item()/float(len(val_loader))
    print("Average UfM latency: " + str(np.average(multiview_times)))
    if write_to_file:
        os.makedirs(os.path.dirname(os.path.join(output_directory,"ensemble_UfM")), exist_ok=True)
        test_csv = os.path.join(output_directory,"ensemble_UfM", "test.csv")
        write_stats_to_file(test_csv=test_csv, avg=avg, avg_NLL=avg_NLL, network_times=network_times, method_times=method_times, multiview_times=multiview_times)
    return

def validate_MCDropout(run_parameters, val_loader, models, criterion, write_to_file=True):
    print("Validating MC-Dropout:")
    global output_directory
    average_meter = AverageMeter() # initialize avg result metric
    # set up model in evaluation mode
    model = models[0] # select first model to test 
    if run_parameters['device'] == 'CPU':
        model.cpu() # switch to CPU
    model.eval() # switch to evaluation mode
    # intialize time and avg metrics
    network_times = []
    method_times = []
    avg_NLL_sum = 0
    # run method for all frames in validation video sequence
    for i, metadata in enumerate(val_loader):
        # load in data 
        input = metadata['rgb']
        target = metadata['depth']
        tic_data = time.perf_counter()
        input = input.cuda(0)
        target = target.cuda(0)
        torch.cuda.synchronize(device=torch.device("cuda:0"))
        data_time = time.perf_counter() - tic_data
        tic_gpu = time.perf_counter()
        tic_method = time.perf_counter()
        # run DNN on this frame
        with torch.no_grad():
            tic_network = time.perf_counter()
            mcdropout_mean_sum, aleatoric_unc_sum = 0, 0 # initialize sums for mean calculation 
            for m in range(run_parameters['num_inferences']):
                output = run_forward_inference(run_parameters, model, input) # model(input) # dropout M times                        
                pred = parse_prediction(output) # single pass predicted depth             
                if m == 0: # initialize old pred
                    old_mcdropout_mean = torch.zeros(pred.shape, device=torch.device("cuda:0"))
                    epistemic_unc = torch.zeros(output[:,0,:,:].shape, device=torch.device("cuda:0"))
                # compute average depth prediction 
                mcdropout_mean_sum = mcdropout_mean_sum + pred
                mcdropout_mean = mcdropout_mean_sum/(m+1.0)
                # compute average aleatoric uncertainty
                aleatoric_unc_sum = aleatoric_unc_sum + parse_aleatoric_uncertainty(output)
                # compute variance of predictions 
                epistemic_unc = compute_iterative_epistemic_uncertainty(pred, epistemic_unc, mcdropout_mean, old_mcdropout_mean, (m+1.0))
                old_mcdropout_mean = mcdropout_mean
            aleatoric_unc = aleatoric_unc_sum/run_parameters['num_inferences']
            torch.cuda.synchronize(device="cuda:0")
            toc_network = time.perf_counter()
            if i > 40: # warm start
                network_times.append(toc_network-tic_network)
        torch.cuda.synchronize(device="cuda:0")
        toc_method = time.perf_counter()
        if i > 40: # warm start 
            method_times.append(toc_method-tic_method)
        gpu_time = time.perf_counter() - tic_gpu
        # measure accuracy and record loss
        result = Result()
        loss = criterion(output, target) # just for logging into result
        result.evaluate(mcdropout_mean.data, target.data, loss)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        # compute NLL 
        NLL = compute_NLL(mcdropout_mean, aleatoric_unc + epistemic_unc, target)
        avg_NLL_sum = avg_NLL_sum + torch.mean(NLL)
        # save files
        if ((i < run_parameters['max_frames_to_save']) or ((i % run_parameters['store_freq'] == 0) and run_parameters['max_frames_to_save'] != 0)):
            output_uncertainty_dir = output_directory+"/MCDropout/per_frame_data/"+str(i)
            save_files(output_uncertainty_dir, gt_np_array=target.data.cpu().data.numpy(), rgb_np_array=np.squeeze(input.data.cpu().data.numpy())[0:3,:,:], pred_np_array=np.squeeze(mcdropout_mean.data.cpu().data.numpy()), aleatoric_np_array=np.squeeze(aleatoric_unc.data.cpu().data.numpy()), epistemic_np_array=np.squeeze(epistemic_unc.data.cpu().data.numpy()), multiview_np_array=np.zeros((224,224)), error_np_array=np.squeeze((target-mcdropout_mean).abs().data.cpu().data.numpy()), NLL_np_array=np.squeeze(NLL.data.cpu().data.numpy()), rgb_img_np_array=np.squeeze(input.data.cpu().data.numpy()).transpose(1,2,0)*255)
        if (i+1) % run_parameters['print_freq'] == 0:
            print_iteration_stats(i, total_val=len(val_loader), gpu_time=gpu_time, result=result, average_meter=average_meter)
    avg = average_meter.average()
    print_avg_stats(avg)
    avg_NLL = avg_NLL_sum.item()/float(len(val_loader))
    if write_to_file:
        os.makedirs(os.path.dirname(os.path.join(output_directory,"MCDropout")), exist_ok=True)
        test_csv = os.path.join(output_directory,"MCDropout", "test.csv")
        write_stats_to_file(test_csv=test_csv, avg=avg, avg_NLL=avg_NLL, network_times=network_times, method_times=method_times, multiview_times=np.asarray([0]))
    return

def validate_full_MCDropout_UfM(run_parameters, val_loader, models, criterion, write_to_file=True):
    print("Validating full MCDropout-UfM:")
    global output_directory
    average_meter = AverageMeter() # initialize avg result metric
    # set up model in evaluation mode
    model = models[0] # select first model to test 
    if run_parameters['device'] == 'CPU':
        model.cpu() # switch to CPU
    model.eval() # switch to evaluation mode
    # intialize time and avg metrics
    network_times = []
    method_times = []
    avg_NLL_sum = 0
    # initialize UfM parameters
    multiview_times = []
    PC = UncertaintyFromMotion(run_parameters, torch.device('cuda:0'))  # initialize point cloud
    # run method for all frames in validation video sequence
    for i, metadata in enumerate(val_loader):
        # load in data 
        input = metadata['rgb']
        target = metadata['depth']
        # move input/target to correct GPU
        tic_data = time.perf_counter()
        input = input.cuda(0)
        target = target.cuda(0)
        torch.cuda.synchronize(device=torch.device("cuda:0"))
        data_time = time.perf_counter() - tic_data
        # run DNN on this frame
        tic_gpu = time.perf_counter()
        tic_method = time.perf_counter()
        with torch.no_grad():
            tic_network = time.perf_counter()
            mcdropout_mean_sum, aleatoric_unc_sum = 0, 0 # initialize sums for mean calculation 
            for m in range(run_parameters['num_inferences']):
                output = run_forward_inference(run_parameters, model, input) # model(input) # dropout M times 
                pred = parse_prediction(output) # single pass predicted depth   
                if m == 0: # initialize old pred
                    old_mcdropout_mean = torch.zeros(pred.shape, device=torch.device("cuda:0"))
                    epistemic_unc = torch.zeros(output[:,0,:,:].shape, device=torch.device("cuda:0"))
                # compute average depth prediction 
                mcdropout_mean_sum = mcdropout_mean_sum + pred
                mcdropout_mean = mcdropout_mean_sum/(m+1.0)
                # compute average aleatoric uncertainty
                aleatoric_unc_sum = aleatoric_unc_sum + parse_aleatoric_uncertainty(output)
                # compute variance of depth predictions 
                epistemic_unc = compute_iterative_epistemic_uncertainty(pred, epistemic_unc, mcdropout_mean, old_mcdropout_mean, (m+1.0))
                old_mcdropout_mean = mcdropout_mean
            aleatoric_unc = aleatoric_unc_sum/run_parameters['num_inferences']
            torch.cuda.synchronize(device="cuda:0")
            toc_network = time.perf_counter()
            if i > 40: # warm start
                network_times.append(toc_network-tic_network)
        torch.cuda.synchronize(device="cuda:0")
        toc_method = time.perf_counter()
        if i > 40: # warm start 
            method_times.append(toc_method-tic_method)
        gpu_time = time.perf_counter() - tic_gpu
        ## begin UfM
        # load rotation and translation
        translation = torch.load(run_parameters['pose_path'] +"/"+str(i)+"/translation.pt", map_location=torch.device('cuda:0'))
        rotation = torch.load(run_parameters['pose_path'] +"/"+str(i)+"/rotation.pt", map_location=torch.device('cuda:0'))
        # run multiview uncertainty
        tic_multiview = time.perf_counter()
        multiview_uncertainty = run_UfM_on_frame(mcdropout_mean, aleatoric_unc+epistemic_unc, translation, rotation, PC, i)
        torch.cuda.synchronize(device="cuda:0")
        toc_multiview = time.perf_counter()
        if i > 40: # warm start
            multiview_times.append(toc_multiview-tic_multiview)
        ## end UfM
        # measure accuracy and record loss
        result = Result()
        loss = criterion(output, target) # just for logging into result
        result.evaluate(mcdropout_mean.data, target.data, loss)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        # compute NLL
        NLL = compute_NLL(mcdropout_mean, multiview_uncertainty, target)
        avg_NLL_sum = avg_NLL_sum + torch.mean(NLL)
        # save files
        if ((i < run_parameters['max_frames_to_save']) or ((i % run_parameters['store_freq'] == 0) and run_parameters['max_frames_to_save'] != 0)):
            output_uncertainty_dir = output_directory+"/full_MCDropout_UfM/per_frame_data/"+str(i)
            save_files(output_uncertainty_dir, gt_np_array=target.data.cpu().data.numpy(), rgb_np_array=np.squeeze(input.data.cpu().data.numpy())[0:3,:,:], pred_np_array=np.squeeze(mcdropout_mean.data.cpu().data.numpy()), aleatoric_np_array=np.squeeze(aleatoric_unc.data.cpu().data.numpy()), epistemic_np_array=np.squeeze(epistemic_unc.data.cpu().data.numpy()), multiview_np_array=np.squeeze(multiview_uncertainty.data.cpu().data.numpy()), error_np_array=np.squeeze((target-mcdropout_mean).abs().data.cpu().data.numpy()), NLL_np_array=np.squeeze(NLL.data.cpu().data.numpy()), rgb_img_np_array=np.squeeze(input.data.cpu().data.numpy()).transpose(1,2,0)*255)
        if (i+1) % run_parameters['print_freq'] == 0:
            print_iteration_stats(i, total_val=len(val_loader), gpu_time=gpu_time, result=result, average_meter=average_meter)
    avg = average_meter.average()
    print_avg_stats(avg)
    avg_NLL = avg_NLL_sum.item()/float(len(val_loader))
    print("Average UfM latency: " + str(np.average(multiview_times)))
    if write_to_file:
        os.makedirs(os.path.dirname(os.path.join(output_directory,"full_MCDropout_UfM")), exist_ok=True)
        test_csv = os.path.join(output_directory,"full_MCDropout_UfM", "test.csv")
        write_stats_to_file(test_csv=test_csv, avg=avg, avg_NLL=avg_NLL, network_times=network_times, method_times=method_times, multiview_times=multiview_times)
    return

def validate_MCDropout_UfM(run_parameters, val_loader, models, criterion, write_to_file=True):
    print("Validating MCDropout-UfM:")
    global output_directory
    average_meter = AverageMeter() # initialize avg result metric
    # set up model in evaluation mode
    model = models[0] # select first model to test 
    if run_parameters['device'] == 'CPU':
        model.cpu() # switch to CPU
    model.eval() # switch to evaluation mode
    # intialize time and avg metrics
    network_times = []
    method_times = []
    avg_NLL_sum = 0
    # initialize UfM parameters
    multiview_times = []
    PC = UncertaintyFromMotion(run_parameters, torch.device('cuda:0'))  # initialize point cloud
    # run method for all frames in validation video sequence
    for i, metadata in enumerate(val_loader):
        # load in data 
        input = metadata['rgb']
        target = metadata['depth']
        tic_data = time.perf_counter()
        input = input.cuda(0)
        target = target.cuda(0)
        torch.cuda.synchronize(device=torch.device("cuda:0"))
        data_time = time.time() - tic_data
        # run DNN on this frame
        tic_gpu = time.time()
        tic_method = time.perf_counter()
        with torch.no_grad():
            tic_network = time.perf_counter()
            output = run_forward_inference(run_parameters, model, input) # model(input) # force one inference
            torch.cuda.synchronize(device="cuda:0")
            toc_network = time.perf_counter()
            if i > 40: # warm start
                network_times.append(toc_network-tic_network)
            # parse depth prediction, aleatoric uncertainty, epistemic uncertainty (if present) from DNN output
            aleatoric_unc = parse_aleatoric_uncertainty(output)
            pred = parse_prediction(output) # single pass predicted depth
            epistemic_unc = torch.zeros(output[:,0,:,:].shape, device=torch.device('cuda:0'))
        torch.cuda.synchronize(device="cuda:0")
        toc_method = time.perf_counter()
        if i > 40: # warm start
            method_times.append(toc_method-tic_method)
        gpu_time = time.time() - tic_gpu
        ## begin UfM
        # load rotation and translation
        translation = torch.load(run_parameters['pose_path'] +"/"+str(i)+"/translation.pt", map_location=torch.device('cuda:0'))
        rotation = torch.load(run_parameters['pose_path'] +"/"+str(i)+"/rotation.pt", map_location=torch.device('cuda:0'))
        # run multiview uncertainty
        tic_multiview = time.perf_counter()
        multiview_uncertainty = run_UfM_on_frame(pred, aleatoric_unc, translation, rotation, PC, i)
        torch.cuda.synchronize(device="cuda:0")
        toc_multiview = time.perf_counter()
        if i > 40: # warm start
            multiview_times.append(toc_multiview-tic_multiview)
        ## end UfM
        # measure accuracy and record loss
        result = Result()
        loss = criterion(output, target) # just for logging into result
        result.evaluate(pred.data, target.data, loss)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        # compute NLL 
        NLL = compute_NLL(pred, multiview_uncertainty, target)
        avg_NLL_sum = avg_NLL_sum + torch.mean(NLL)
        # save files
        if ((i < run_parameters['max_frames_to_save']) or ((i % run_parameters['store_freq'] == 0) and run_parameters['max_frames_to_save'] != 0)):
            output_uncertainty_dir = output_directory+"/MCDropout_UfM/per_frame_data/"+str(i)
            save_files(output_uncertainty_dir, gt_np_array=target.data.cpu().data.numpy(), rgb_np_array=np.squeeze(input.data.cpu().data.numpy())[0:3,:,:], pred_np_array=np.squeeze(pred.data.cpu().data.numpy()), aleatoric_np_array=np.squeeze(aleatoric_unc.data.cpu().data.numpy()), epistemic_np_array=np.squeeze(epistemic_unc.data.cpu().data.numpy()), multiview_np_array=np.squeeze(multiview_uncertainty.data.cpu().data.numpy()), error_np_array=np.squeeze((target-pred).abs().data.cpu().data.numpy()), NLL_np_array=np.squeeze(NLL.data.cpu().data.numpy()), rgb_img_np_array=np.squeeze(input.data.cpu().data.numpy()).transpose(1,2,0)*255)
        if (i+1) % run_parameters['print_freq'] == 0:
            print_iteration_stats(i, total_val=len(val_loader), gpu_time=gpu_time, result=result, average_meter=average_meter)
    avg = average_meter.average()
    print_avg_stats(avg)
    avg_NLL = avg_NLL_sum.item()/float(len(val_loader))
    print("Average UfM latency: " + str(np.average(multiview_times)))
    if write_to_file:
        os.makedirs(os.path.dirname(os.path.join(output_directory,"MCDropout_UfM")), exist_ok=True)
        test_csv = os.path.join(output_directory,"MCDropout_UfM", "test.csv")
        write_stats_to_file(test_csv=test_csv, avg=avg, avg_NLL=avg_NLL, network_times=network_times, method_times=method_times, multiview_times=multiview_times)
    return

def parse_prediction(output):
    # for aleatoric uncertatinty, evidential regression, l1/l2 loss
    return torch.reshape(output[:,0,:,:],(output.shape[0],1,output.shape[2],output.shape[3]))

def parse_aleatoric_uncertainty(output):
    if run_parameters['loss'] == "heteroscedastic": # aleatoric loss
        log_var = torch.reshape(output[:,1,:,:],(output.shape[0],1,output.shape[2],output.shape[3]))
        aleatoric_unc = torch.exp(log_var)
    else:
        aleatoric_unc = torch.zeros(output[:,0,:,:].shape, device=torch.device('cuda:0'))
    return aleatoric_unc

def compute_iterative_epistemic_uncertainty(pred, epistemic_unc, new_mean, old_mean, n):
    Sn = epistemic_unc*(n-1) # previous Sn 
    Sn = Sn + (pred - old_mean)*(pred - new_mean)
    epistemic_unc = Sn/n # variance
    return epistemic_unc
def compute_NLL(pred, total_uncertainty, target):
    # transfer to same GPU
    total_uncertainty = torch.squeeze(total_uncertainty.cuda(0))
    nonzero_uncertainty_idx = total_uncertainty > 1e-10
    NLL = 0.5*torch.log(2*math.pi*(total_uncertainty[nonzero_uncertainty_idx])) + torch.mul(1.0/(2*total_uncertainty[nonzero_uncertainty_idx]),torch.pow((pred[:,:,nonzero_uncertainty_idx].cuda(0)-target[:,:,nonzero_uncertainty_idx].cuda(0)),2)) 
    return NLL 

if __name__ == '__main__':
    run_method()

