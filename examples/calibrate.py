import numpy as np
import matplotlib.pyplot as plt
from uncertainty_from_motion.libs.uncertainty_calibration.src.calibration_curve import CalibrationCurve
import uncertainty_from_motion.libs.uncertainty_calibration.src.plotting as plotting 
import os
from pprint import pprint
import time
import sys
import csv  

# parameters 
uncertainty_type = sys.argv[1]
results_dnn_dir = sys.argv[2]
results_calibration_dir = sys.argv[3]
delta_interval = 0.25 # (1+delta = delta_x metric, set equal to 0.25 for original delta1 metric)
warm_start = 0 # number of frames to ignore at the start of the calibration (set equal to 0 to analyze all frames)
frame_start = 0 # starting frame idx 
frame_end = -1 # either ending frame idx or full sequence 
# calculate calibration curves 
CC = CalibrationCurve(uncertainty_type, directory=results_dnn_dir, output_directory=results_calibration_dir, frame_start = frame_start, frame_end = frame_end, delta=delta_interval, warm_start=warm_start)
confidence_accuracy_bins, var_sq_error_bins, ppf_bins, accuracy = CC.calculate_metrics()

# calculate average metrics from curves 
ECE_conf_acc = CC.calculate_ece(confidence_accuracy_bins)
RMSE_conf_acc = CC.calculate_rmse(confidence_accuracy_bins)
ECE_ppf = CC.calculate_ece(ppf_bins)
RMSE_ppf = CC.calculate_rmse(ppf_bins)
ECE_var_sq_error = CC.calculate_ece(var_sq_error_bins)
RMSE_var_sq_error = CC.calculate_rmse(var_sq_error_bins)

# store results 
if not os.path.exists(results_calibration_dir):
    os.makedirs(results_calibration_dir)
results_calibration_csv = os.path.join(results_calibration_dir, 'calibration_stats.csv')
# create new csv files with only header
fieldnames = ['delta_interval','accuracy','ECE_conf_acc','RMSE_conf_acc','ECE_ppf','RMSE_ppf','ECE_var_sq_error','RMSE_var_sq_error']
with open(results_calibration_csv, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'delta_interval': delta_interval, 'accuracy': accuracy, 'ECE_conf_acc': ECE_conf_acc, 'RMSE_conf_acc': RMSE_conf_acc, 'ECE_ppf': ECE_ppf, 'RMSE_ppf': RMSE_ppf, 'ECE_var_sq_error': ECE_var_sq_error, 'RMSE_var_sq_error': RMSE_var_sq_error})

# plot calibration curves 
title_fig = f'Acc. vs. conf., Delta = {delta_interval:.2f}, accuracy = {accuracy:.3f}, ECE = {ECE_conf_acc:.3f}, RMSE = {RMSE_conf_acc:.3f}'
fig = plotting.plot_bins(confidence_accuracy_bins, results_calibration_dir, delta_interval, title_fig, 'Confidence (delta-1)', 'Delta-1 Accuracy',x_save="confidence",y_save="delta_1_accuracy", ERROR_BARS=False, BIN_COUNT=True, SAVE_FIG=True)
title_fig_2 = f'Var. vs. squared error, ECE = {ECE_var_sq_error:.3f}, RMSE = {RMSE_var_sq_error:.3f}'
fig_2 = plotting.plot_bins(var_sq_error_bins, results_calibration_dir, delta_interval, title_fig_2, 'Variance', 'Avg. squared error', x_save="variance", y_save="avg_sq_error", ERROR_BARS=False, BIN_COUNT=False, SAVE_FIG=True)
title_fig_3 = f'Confidence vs. Frequency (ppf), ECE = {ECE_ppf:.3f}, RMSE = {RMSE_ppf:.3f}'
fig_3 = plotting.plot_bins(ppf_bins, results_calibration_dir, delta_interval, title_fig_3, 'Confidence', 'Frequency (ppf)', x_save="confidence", y_save="frequency_ppf", ERROR_BARS=False, BIN_COUNT=False, SAVE_FIG=True)
plt.close()