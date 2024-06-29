'''
Name: run_UfM_on_frame
Description: Given depth prediction, aleatoric uncertainty prediction from DNN, pose, and point cloud from previous frames, run Uncertainty from Motion (UfM) and return UfM uncertainty for current frame 
Inputs:
    prediction     : depth prediction tensor from DNN output 
    uncertainty    : aleatoric prediction tensor from DNN output
    run_parameters : run_parameters for this experiment (defined in utils)
    translation    : translation tensor 
    rotation       : rotation tensor
    PC_compressed  : point cloud (covariance in compressed form) computed by UfM on last frame in sequence
Output: 
    frame_uncertainty : UfM uncertainty
'''

def run_UfM_on_frame(prediction, uncertainty, translation, rotation, PC_compressed, frame_num):
        if (frame_num == 1):
            PC_compressed.first_frame = False # turn off first frame flag
        frame_uncertainty = PC_compressed.add_depth_frame_to_point_cloud(prediction, translation, rotation, uncertainty=uncertainty)
        return frame_uncertainty


