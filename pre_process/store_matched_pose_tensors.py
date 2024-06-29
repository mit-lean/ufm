import torch 
import numpy as np 
import quaternion
import os 
import sys 

# find_timestamp_for_frame: find timestamp corresponding to frame number, assuming
#                        we have TUM formatted directory and no timestamp in the
#                        name of the frame
def find_timestamp_for_frame(frame_number, original_sequence_directory):
    stamped_dir_path = original_sequence_directory + "/rgb"
    # find timestamps from timestamp directory
    timestamp_list = []
    files_int = []
    for subdir, dirs, files in os.walk(stamped_dir_path):
        for f in files:
            files_int.append(f[:-4])
        for file in sorted(list(map(float,files_int))):
            timestamp_list.append(file)
    return(timestamp_list[int(frame_number)])

def find_timestamp_from_associations_file(frame_number, asso_filepath):
    with open(asso_filepath, "r") as associations_file:
        lines = associations_file.readlines()
        frame_line = lines[frame_number]
        timestamp = float(frame_line[:17])
    return timestamp

def match_pose(frame_number, asso_filepath, pose_filepath, timestamp_factor=1.0):
    # convert pose text file into np array
    pose_array = np.loadtxt(pose_filepath)
    # find timestamp corresponding to frame number
    timestamp = find_timestamp_from_associations_file(frame_number, asso_filepath)
    timestamp = timestamp/timestamp_factor # unit conversion
    # find and return pose closest to timestamp
    if "Monocular" in pose_filepath:
        # extra step of making pose_filepath grab right timestamp from associations file 
        for i in range(pose_array[:,0].shape[0]):
            pose_array[i,0] = find_timestamp_for_frame(int(pose_array[i,0]), original_sequence_directory=asso_filepath[:-16])
    closest_timestamp_row = np.abs(pose_array[:,0]-timestamp).argmin() # find the closest timestamped pose
    pose_row = pose_array[closest_timestamp_row,:] # select that row
    # assemble into quaternion and translation arrays
    t = pose_row[1:4]
    q = pose_row[4:]
    return t,q

def matched_pose_to_torch_files(asso_filepath, pose_filepath, num_files, output_dir, timestamp_factor=1.0, max_frames_to_process=np.inf):
    """ calculate matched poses and save as torch files """
    files = range(0,num_files)
    for i in files:
        if i < max_frames_to_process:
            translation, quat = match_pose(
                    i, asso_filepath, pose_filepath, timestamp_factor)
            # calculate rotation from quaternion 
            rotation = quaternion.as_rotation_matrix(
                       np.quaternion(quat[3], quat[0], quat[1], quat[2]))
            quaternion_np = np.asarray((quat[3], quat[0], quat[1], quat[2]))
            # convert numpy array to torch
            translation_tensor = torch.from_numpy(translation)
            rotation_tensor = torch.from_numpy(rotation)
            quaternion_tensor = torch.from_numpy(quaternion_np)
            # save torch file 
            if not os.path.exists(output_dir +"/"+str(i)):
                os.makedirs(output_dir +"/"+str(i))
            torch.save(translation_tensor, output_dir+"/"+str(i)+"/translation.pt")
            torch.save(rotation_tensor, output_dir+"/"+str(i)+"/rotation.pt")
            torch.save(quaternion_tensor, output_dir+"/"+str(i)+"/quaternion.pt")
            # save csv file 
            np.savetxt(output_dir+"/"+str(i)+"/translation.csv", translation, delimiter=",")
            np.savetxt(output_dir+"/"+str(i)+"/rotation.csv", rotation, delimiter=",")
            np.savetxt(output_dir+"/"+str(i)+"/quaternion.csv", quaternion_np, delimiter=",")
            # save numpy file
            np.save(output_dir+"/"+str(i)+"/translation.npy", translation)
            np.save(output_dir+"/"+str(i)+"/rotation.npy", rotation)
            np.save(output_dir+"/"+str(i)+"/quaternion.npy", quaternion_np)
    return 

def calculate_num_files(path_to_num_files):
    with open(path_to_num_files, "r") as associations_file: # only for associations.py format
        lines = associations_file.readlines()
    return len(lines)


if __name__ == '__main__':
    # sys.argv[1]: path to associations.py
    # sys.argv[2]: path to pose text file
    # sys.argv[3]: path to output directory for stored poses
    num_files = calculate_num_files(sys.argv[1])
    matched_pose_to_torch_files(sys.argv[1], sys.argv[2], num_files, sys.argv[3], timestamp_factor=1.0, max_frames_to_process=np.inf)
