from array import array
import numpy as np
import matplotlib.pyplot as plt 
import sys 
import os 
import subprocess
import time
import pandas as pd
import yaml 
from glob import glob
import matplotlib.style as mplstyle
mplstyle.use('fast')
from PIL import Image # burn in text
from PIL import ImageDraw # burn in text
from PIL import ImageFont # burn in text

def colored_depthmap(depth, cmap, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return cmap(depth_relative)[:,:,:3] # H, W, C cmap = plt.cm.viridis

def load_arrays(path_to_results_dir, subdir_num, total_unc_type):
    # load in RGB 
    rgb = np.load(os.path.join(path_to_results_dir, str(subdir_num), "rgb_input.npy"))
    # rgb = np.rint(rgb).astype(int)
    rgb = np.transpose(np.squeeze(rgb), (1,2,0)) # H, W, C
    # load in ground truth 
    gt = np.squeeze(np.load(os.path.join(path_to_results_dir, str(subdir_num), "gt.npy")))
    # load in depth prediction 
    pred = np.squeeze(np.load(os.path.join(path_to_results_dir, str(subdir_num), "pred.npy")))
    # load in error
    error = np.squeeze(np.load(os.path.join(path_to_results_dir, str(subdir_num), "error.npy")))
    error[gt == 0] = 0 # mask out missing gt data
    # load in total uncertainty 
    # load in aleatoric unc 
    aleatoric_unc = np.squeeze(np.load(os.path.join(path_to_results_dir, str(subdir_num), "aleatoric_uncertainty.npy")))
    # load in epistemic unc 
    epistemic_unc = np.squeeze(np.load(os.path.join(path_to_results_dir, str(subdir_num), "epistemic_uncertainty.npy")))
    # load in multiview unc
    multiview_unc = np.squeeze(np.load(os.path.join(path_to_results_dir, str(subdir_num), "multiview_uncertainty.npy")))
    if total_unc_type == "epistemic_plus_aleatoric":
        total_unc = aleatoric_unc+epistemic_unc
    elif total_unc_type == "multiview":
        total_unc = multiview_unc
    # load in NLL
    nll = np.squeeze(np.load(os.path.join(path_to_results_dir, str(subdir_num), "NLL.npy")))
    nll[gt == 0] = 0 # mask out missing gt data
    return [rgb, gt, pred, error, total_unc, nll]

def initialize_visualizer(height, width, path_to_results_dir):
    # animating figure for visualization
    fig, ax = plt.subplots(1,6, figsize=(40,5))
    # initialize with random data to start 
    ax[0].set_title('RGB input')
    ax[1].set_title('Ground-truth depth')
    ax[2].set_title('DNN prediction')
    ax[3].set_title('Error')
    ax[4].set_title('Uncertainty')
    ax[5].set_title('NLL')
    im1 = ax[0].imshow(3*np.random.rand(height, width), aspect='auto', cmap = 'viridis')
    im2 = ax[1].imshow(3*np.random.rand(height, width), aspect='auto', cmap = 'viridis')
    im3 = ax[2].imshow(3*np.random.rand(height, width), aspect='auto', cmap = 'viridis')
    im4 = ax[3].imshow(3*np.random.rand(height, width), aspect='auto', cmap = 'inferno')
    im5 = ax[4].imshow(3*np.random.rand(height, width), aspect='auto', cmap = 'inferno')
    im6 = ax[5].imshow(3*np.random.rand(height, width), aspect='auto', cmap = 'inferno')
    # turn of xticks/yticks for images
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    ax[5].set_xticks([])
    ax[5].set_yticks([])
    return fig, ax, im1, im2, im3, im4, im5, im6

def update_images(arrays, gt_min_max, pred_min_max, error_min_max, unc_min_max, nll_min_max, im1, im2, im3, im4, im5, im6):
    # arrays: [rgb, gt, pred, error, total unc, nll]
    # color consistent for each image 
    gt_col = colored_depthmap(arrays[1], plt.cm.viridis, gt_min_max[2], gt_min_max[3])
    pred_col = colored_depthmap(arrays[2], plt.cm.viridis, gt_min_max[2], gt_min_max[3])
    error_col = colored_depthmap(arrays[3], plt.cm.inferno, error_min_max[0], error_min_max[3])
    unc_col = colored_depthmap(arrays[4], plt.cm.inferno, unc_min_max[0], unc_min_max[3])
    nll_col = colored_depthmap(arrays[5], plt.cm.inferno, nll_min_max[0], nll_min_max[3])
    # alternative: comment in to hold constant for consistent comparison across methods
    # error_col = colored_depthmap(arrays[3], plt.cm.inferno, 0, 1.5)
    # unc_col = colored_depthmap(arrays[4], plt.cm.inferno, 0, 1.0)
    # nll_col = colored_depthmap(arrays[4], plt.cm.inferno, 0, 1.5)
    im1.set_data(arrays[0]) # rgb 
    im2.set_data(gt_col)
    im3.set_data(pred_col)
    im4.set_data(error_col)
    im5.set_data(unc_col)
    im6.set_data(nll_col)
    return im1, im2, im3, im4, im5, im6
    
def get_global_min_max(data_list):
    # concatenate all numpy arrays in directory 
    data_concat = np.concatenate(data_list)
    min_pred = np.min(data_concat)
    max_pred = np.max(data_concat)
    min_99_quantile = np.quantile(data_concat,0.01)
    max_99_quantile = np.quantile(data_concat,0.99)
    return [min_pred, max_pred, min_99_quantile, max_99_quantile]

def get_min_max_vals(path_to_results_dir, total_unc_type, idx_list = None):
    # initialize containers for consistent colormaps
    gts = []
    preds = []
    errors = []
    uncs = []
    nlls = []
    # calculate consistent color
    print("calculating max and min values for consistent colormaps")
    if idx_list == None:
        idx_list = sorted(list(map(int,os.listdir(path_to_results_dir))))
    for num_subdir in idx_list: # convert to int numbers for sorting
        # load in arrays
        arrays = load_arrays(path_to_results_dir, num_subdir, total_unc_type) # rgb, gt, pred, error, aleatoric_var, epistemic_var, e_plus_a_var, e_minus_a_var 
        gts.append(arrays[1])
        preds.append(arrays[2])
        errors.append(arrays[3])
        uncs.append(arrays[4])
        nlls.append(arrays[5])
    gt_min_max = get_global_min_max(gts) # min_gt, max_gt, min_99_quantile_gt, max_99_quantile_gt
    pred_min_max = get_global_min_max(preds) # min_pred, max_pred, min_99_quantile_pred, max_99_quantile_pred
    error_min_max = get_global_min_max(errors) # min_error, max_error, min_99_quantile_error, max_99_quantile_error
    unc_min_max = get_global_min_max(uncs) # min_aleatoric_var, max_aleatoric_var, min_01_quantile_aleatoric_var, max_99_quantile_aleatoric_var
    nll_min_max = get_global_min_max(nlls) # min_aleatoric_var, max_aleatoric_var, min_01_quantile_aleatoric_var, max_99_quantile_aleatoric_var
    print("error_min_max")
    print(error_min_max)
    print("unc_min_max")
    print(unc_min_max)
    print("nll_min_max")
    print(nll_min_max)
    return gt_min_max, pred_min_max, error_min_max, unc_min_max, nll_min_max

def make_video(output_folder, vid_name = "visualization_video"):
    os.chdir(output_folder)
    subprocess.call([
        'ffmpeg', '-framerate', '30', '-i', '%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        vid_name + '.mp4'
    ])
    for file_name in glob("*.png"):
        os.remove(file_name)
    return 

def visualize_frames(path_to_results_dir, video_dir, height, width, total_unc_type):
    # initialize visualizer
    fig, ax, im1, im2, im3, im4, im5, im6 = initialize_visualizer(height, width, path_to_results_dir)
    # get min/max values for consistent color for images
    gt_min_max, pred_min_max, error_min_max, unc_min_max, nll_min_max = get_min_max_vals(path_to_results_dir + "/per_frame_data/", total_unc_type)
    # iterate through frames in results directory 
    for num_subdir in sorted(list(map(int,os.listdir(path_to_results_dir + "per_frame_data/")))): # convert to int numbers for sorting
        # load in arrays
        if (num_subdir % 100 == 0): 
            print("Visualizing frame " + str(num_subdir))
        arrays = load_arrays(path_to_results_dir + "per_frame_data/", num_subdir, total_unc_type) # rgb, gt, pred, error, total unc, nll
        im1, im2, im3, im4, im5, im6 = update_images(arrays, gt_min_max, pred_min_max, error_min_max, unc_min_max, nll_min_max, im1, im2, im3, im4, im5, im6)
        fig.canvas.draw()
        if (video_dir != "False"):
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            plt.savefig(video_dir + "/%02d.png" % num_subdir) # for video 
            # add burn in text to that image adding seq_idx number 
            img_no_burn_txt = Image.open(video_dir + "/%02d.png" % num_subdir)
            # font style and font size
            FreeMono_font = ImageFont.truetype('FreeMono.ttf', 32)
            img_drawer = ImageDraw.Draw(img_no_burn_txt)
            # add seq_idx text to an image
            img_drawer.text((28, 36), "seq_idx: " + str(num_subdir), font = FreeMono_font, fill=(0, 0, 0))
            # add nll text to an image 
            ## compute average nll 
            avg_nll = round(np.average(arrays[5]),2)
            img_drawer.text((28, 75), "NLL: " + str(avg_nll), font = FreeMono_font, fill=(0, 0, 0))
            img_no_burn_txt.save(video_dir + "/%02d.png" % num_subdir)
        plt.pause(0.0001)
        fig.canvas.flush_events()  
    if (video_dir != "False"):
        make_video(output_folder = video_dir, vid_name = "visualization_video")
    return

if __name__ == '__main__':
    visualize_frames(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
