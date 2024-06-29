import numpy as np
import time
import torch

'''
Class name            : UncertaintyFromMotion
Description           : Runs UfM algorithm
Initialization inputs
    run_parameters    : run_parameters for this experiment (defined in config file)
    cuda0             : GPU to run code on 
Methods
    add_depth_frame_to_point_cloud
    rotate_point_cloud_to_cam_frame_relative
    reproject_point_cloud_to_frame
    rotate_variance_frame
    project_frame_to_point_cloud
    update_point_cloud_optimized
    vec_squared
    compress_symmetric_matrix
    uncompress_compressed_symmetric_matrix
    uncompress_compressed_symmetric_matrix_flattened
    uncompress_compressed_matrix_flattened_vert
    compress_vertical_array
    stack_horizontal_array_by_entry_not_symmetric
    calculate_masked_uncertainty
    delete_points_from_point_cloud
'''

class UncertaintyFromMotion:
    def __init__(self, run_parameters, cuda0):
        self.cuda = cuda0
        self.counter_cloud = torch.empty((1,1), device=self.cuda)
        self.pos_cloud = torch.empty((3,1), device=self.cuda)
        self.covar_cloud = torch.empty((6,1), device=self.cuda)
        # # preprocessing/cutting of image dimensions
        self.height = run_parameters['input_height'] 
        self.width = run_parameters['input_width'] 
        self.num_of_pixels = self.height*self.width
        # camera calibration 
        self.fx = run_parameters['fx']
        self.fy = run_parameters['fy']
        self.cx = run_parameters['cx']
        self.cy = run_parameters['cy']
        # flags 
        self.first_frame = True
        self.max_cloud_size = run_parameters['max_cloud_size']
        # constant pixel arrays indicated row or col
        # initialize a u (pixel) array where u is index of col (total = total col)
        self.u = torch.zeros((self.height,self.width), device=self.cuda) # (H,W)
        for i in range(self.width): # starts from 0
            self.u[:,i] = i
        # initialize a v (pixel) array where v is index of row (total = total row)
        self.v = torch.zeros((self.height,self.width), device=self.cuda) # (H,W)
        for j in range(self.height):
            self.v[j,:] = j
        # last rotation, translation 
        self.last_rotation = torch.zeros((3,3), device=self.cuda)
        self.last_translation = torch.zeros((3,1), device=self.cuda)
        # initialize number of points deleted from point cloud
        self.number_pts_deleted_out_of_view, self.number_pts_deleted_oldest = 0, 0
        # initialize timer for deletion time
        self.sum_deletion_time = 0

    '''
    Name: add_depth_frame_to_point_cloud
    Description: Given current frame's depth prediction and aleatoric uncertainty prediction, update UfM point cloud and return UfM uncertainty for current frame 
    Inputs:
        prediction     : depth prediction tensor from DNN output 
        translation    : translation tensor 
        rotation       : rotation tensor
        uncertainty    : aleatoric prediction tensor from DNN output
    Output: 
        frame_uncertainty : UfM uncertainty
    '''
    def add_depth_frame_to_point_cloud(self, pred_depth_np, translation, rotation, uncertainty=None):
        # check which points in point cloud are in RF of current frame
        # initialize pixels in frame as all not in frame
        mask_in_frame = torch.zeros((1, 1), device=self.cuda)
        masked_pixels_in_frame = torch.zeros((self.height, self.width), device=self.cuda)  # (H,W)
        uj = torch.zeros((1, 1), device=self.cuda)  # (1,N), N = # of points in point cloud
        vj = torch.zeros((1, 1), device=self.cuda)  # (1,N), N = # of points in point cloud
        valid_point_indices = torch.empty((1,1), device=self.cuda) # (1, N), N = # of points in point cloud
        invalid_point_indices = torch.empty((1,1), device=self.cuda) # (1, N), N = # of points in point cloud
        valid_idx_img_with_repeats = torch.empty((1,1), device=self.cuda) # (1, N_seen) = # of points in point cloud seen 
        # rotate point cloud to current camera frame 
        ## rotate from last camera reference frame to this camera reference frame
        # calculate relative rotation, translation 
        if self.first_frame:
            rel_rotation = rotation 
            rel_translation = translation
        else:
            rel_rotation = torch.matmul(torch.transpose(rotation, 0, 1), self.last_rotation)
            rel_translation = torch.matmul(torch.transpose(rotation, 0, 1), (self.last_translation-translation))
        self.last_rotation = rotation 
        self.last_translation = translation 
        pos_cloud_cam_rf, covar_cloud_cam_rf = self.rotate_point_cloud_to_cam_frame_relative(rel_translation, rel_rotation)
        # if this isn't the first frame, check if there are points in 3D point cloud we are
        # seeing again in this frame
        if not self.first_frame:
            mask_in_frame, masked_pixels_in_frame, uj, vj, valid_point_indices, valid_idx_img_with_repeats, invalid_point_indices = self.reproject_point_cloud_to_frame(pos_cloud_cam_rf)
        # project pixels in current frame back to point cloud
        frame_uncertainty, pos_cloud_cam_rf_updated, covar_cloud_cam_rf_updated, num_new_points_added = self.project_frame_to_point_cloud(pos_cloud_cam_rf, covar_cloud_cam_rf, pred_depth_np, translation, rotation, mask_in_frame, masked_pixels_in_frame, uj, vj, valid_point_indices, valid_idx_img_with_repeats, uncertainty)
        # store updated point cloud  
        self.pos_cloud = pos_cloud_cam_rf_updated
        self.covar_cloud = covar_cloud_cam_rf_updated
        # delete points from point cloud to stay under point cloud size  
        self.number_pts_deleted_out_of_view, self.number_pts_deleted_oldest = self.delete_points_from_point_cloud(valid_point_indices, invalid_point_indices, num_new_points_added)
        return frame_uncertainty
    
    '''
    Name: rotate_point_cloud_to_cam_frame_relative 
    Description: rotate point cloud to current camera frame from previous camera frame
    Inputs:
        rel_translation     : relative rotation tensor 
        rel_rotation        : relative translation tensor 
    Output: 
        pos_cloud_cam_rf    : position point cloud in current camera reference frame 
        covar_cloud_cam_rf  : covariance point cloud in current camera reference frame
    '''
    def rotate_point_cloud_to_cam_frame_relative(self, rel_translation, rel_rotation):
        pos_cloud_cam_rf = torch.matmul(rel_rotation, self.pos_cloud.double())+torch.reshape(rel_translation, (3, 1))
        num_points_point_cloud = self.pos_cloud.size()[1]
        uncompressed_cov_cloud = self.uncompress_compressed_symmetric_matrix_flattened(a_diag=self.covar_cloud[0,:], b= self.covar_cloud[3,:], c=self.covar_cloud[4,:], \
                            d_diag=self.covar_cloud[1,:], e=self.covar_cloud[5,:], f_diag=self.covar_cloud[2,:], num_of_points=num_points_point_cloud)
        covar_cloud_cam_rf = self.rotate_variance_frame(rel_rotation, uncompressed_cov_cloud, num_of_points=num_points_point_cloud) # (3xHxW,3)
        covar_cloud_cam_rf = self.compress_vertical_array(covar_cloud_cam_rf, num_of_points=num_points_point_cloud)
        return pos_cloud_cam_rf, covar_cloud_cam_rf 
    
    '''
    Name: reproject_point_cloud_to_frame
    Description: find correspondences between point cloud (point we have seen before) and pixels of new image
    Inputs:
        pos_cloud_cam_rf            : position point cloud in current camera reference frame
    Output:
        mask_in_frame               : mask on point cloud array for which points in point cloud are in frame 
        masked_pixels_in_frame      : mask on frame for which pixels of the frame are in the point cloud 
        uj                          : u pixel locations for each point in point cloud
        vj                          : v pixel locations for each point in point cloud
        valid_point_indices         : indices of mask_in_frame that are nonzero 
        valid_idx_img_with_repeats  : indices of image that are nonzero with repeats 
    '''
    def reproject_point_cloud_to_frame(self, pos_cloud_cam_rf):
        # reproject 3D point in cloud to pixel u,v in current image 
        uj = (torch.round(
            (self.fx*pos_cloud_cam_rf[0, :]/pos_cloud_cam_rf[2, :]+self.cx))).int()
        vj = (torch.round(
            (self.fy*pos_cloud_cam_rf[1, :]/pos_cloud_cam_rf[2, :]+self.cy))).int()
        # calculate if valid pixel 
        mask_in_frame = torch.logical_and(vj < self.height, torch.logical_and(vj >= 0, torch.logical_and(uj < self.width,
            torch.logical_and(uj >= 0, pos_cloud_cam_rf[2, :] >= 0))))
        # find index of mask in frame where valid (1) to store pixel-wise mask
        masked_pixels_in_frame = torch.zeros((self.height, self.width), device=self.cuda) # FIXME: initialize somewhere else for efficiency
        valid_point_indices = torch.nonzero(mask_in_frame).long()
        invalid_point_indices = torch.nonzero(mask_in_frame == 0).long()
        valid_idx_img_with_repeats = (vj[valid_point_indices]*self.width + uj[valid_point_indices]).long() # row-major order  
        masked_pixels_in_frame[vj[valid_point_indices].long(),uj[valid_point_indices].long()] = 1
        return mask_in_frame, masked_pixels_in_frame, uj, vj, valid_point_indices, valid_idx_img_with_repeats, invalid_point_indices
            
    
    '''
    Name: rotate_variance_frame
    Description: rotate compressed covariance 
    Inputs:
        rotation                              : rotation tensor
        uncompressed_covariance_matrix        : uncompressed covariance tensor 
        num_of_points                         : number of points in point cloud 
    Output:
        covariance_cam_frame_split_recombined : rotated uncompressed covariance tensor
    '''
    def rotate_variance_frame(self,rotation,uncompressed_covariance_matrix, num_of_points):
        covariance_cam_frame_partway = torch.matmul(rotation,uncompressed_covariance_matrix.double()) # (3,3xHxW)
        # compress to depth wise again 
        compressed_array = self.stack_horizontal_array_by_entry_not_symmetric(covariance_cam_frame_partway, num_of_points)
        # uncompress to vertical stack 
        uncompressed_vert_array = self.uncompress_compressed_matrix_flattened_vert(a_diag=compressed_array[0,:], b=compressed_array[3,:], 
                                        c=compressed_array[4,:], d_diag=compressed_array[1,:], e=compressed_array[5,:], f_diag=compressed_array[2,:],
                                        g=compressed_array[6,:], h=compressed_array[7,:] , i =compressed_array[8,:], num_of_points = num_of_points)
        covariance_cam_frame_split_recombined = torch.matmul(uncompressed_vert_array.double(),torch.transpose(rotation,0,1))
        return covariance_cam_frame_split_recombined

    '''
    Name: project_frame_to_point_cloud
    Description: add current image, updating points in point cloud that we are seeing again and adding points to point cloud we are seeing for the first time
    Inputs:
        pos_cloud_cam_rf            : position point cloud in current camera reference frame
        covar_cloud_cam_rf          : covariance point cloud in current camera reference frame
        pred_depth_np               : depth prediction tensor from DNN output 
        translation                 : translation tensor 
        rotation                    : rotation tensor
        mask_in_frame               : mask on point cloud array for which points in point cloud are in frame 
        masked_pixels_in_frame      : mask on frame for which pixels of the frame are in the point cloud 
        uj                          : u pixel locations for each point in point cloud
        vj                          : v pixel locations for each point in point cloud
        valid_point_indices         : indices of mask_in_frame that are nonzero 
        valid_idx_img_with_repeats  : indices of image that are nonzero with repeats 
    Output:
        frame_uncertainty           : UfM uncertainty
        pos_cloud_cam_rf_updated    : updated position point cloud in current camera reference frame
        covar_cloud_cam_rf_updated  : updated covariance point cloud in current camera reference frame
        num_new_points_added        : number of points seen for the first time added to point cloud 
    '''
    def project_frame_to_point_cloud(self,pos_cloud_cam_rf, covar_cloud_cam_rf, pred_depth_np,translation,rotation,
                                 mask_in_frame, masked_pixels_in_frame,uj,vj, valid_point_indices, valid_idx_img_with_repeats, uncertainty=None):
        pt_cam_rf = torch.zeros((self.height,self.width,3),device=self.cuda) # (H,W,3)
        pt_cam_rf[:,:,2] = pred_depth_np # (H,W,1) z of pixel projection
        pt_cam_rf[:,:,0] = torch.mul((self.u-self.cx),pt_cam_rf[:,:,2])/self.fx # x of pixel point in camera frame
        pt_cam_rf[:,:,1] = torch.mul((self.v-self.cy),pt_cam_rf[:,:,2])/self.fy # y pixel point in camera frame
        # flatten and transpose 3D array to 2D array for easy matrix multiplication
        # (H,W,3) --> (3,HxW)
        pt_cam_rf_flattened = torch.transpose(pt_cam_rf.reshape(-1,pt_cam_rf.size()[-1]), 0, 1) # (3,HxW)
        # for pixels of points that have not been seen before, store in point cloud
        masked_pixels_out_frame = torch.logical_not(masked_pixels_in_frame)  # (H,W)
        masked_pixels_out_frame = torch.flatten(masked_pixels_out_frame)
        pt_not_seen = pt_cam_rf_flattened[:,masked_pixels_out_frame] # (3, HxW) 
        num_new_points_added = pt_not_seen.size()[1] 
        # store the new points to the point cloud (counter, mean 3D world position pred,
        #       3D variance world position predicted, mean ground-truth world position predicted,
        #       variance ground-truth world position, 3D world position pred sample, empty samples)
        if uncertainty is not None:
            # calculate vector for each pixel in camera RF
            covar_cam_rf = torch.zeros((self.height,self.width,1), device=self.cuda) # (H,W,1)
            covar_cam_rf_xx = torch.zeros((self.height,self.width,1), device=self.cuda) # (H,W,1)
            covar_cam_rf_yy = torch.zeros((self.height,self.width,1), device=self.cuda) # (H,W,1)
            covar_cam_rf_xz = torch.zeros((self.height,self.width,1), device=self.cuda) # (H,W,1)
            covar_cam_rf_yz = torch.zeros((self.height,self.width,1), device=self.cuda) # (H,W,1)
            covar_cam_rf_xy = torch.zeros((self.height,self.width,1), device=self.cuda) # (H,W,1)
            covar_cam_rf[:,:,0] = uncertainty # (H,W,1) uncertainty predicted # z-axis
            covar_cam_rf_xx[:,:,0] = torch.mul(torch.pow((self.u-self.cx)/self.fx, 2),uncertainty)
            covar_cam_rf_yy[:,:,0] = torch.mul(torch.pow((self.v-self.cy)/self.fy, 2),uncertainty)
            covar_cam_rf_xz[:,:,0] = torch.mul((self.u-self.cx)/self.fx, uncertainty)
            covar_cam_rf_yz[:,:,0] = torch.mul((self.v-self.cy)/self.fy, uncertainty)
            covar_cam_rf_xy[:,:,0] = torch.mul(torch.mul((self.u-self.cx)/self.fx, (self.v-self.cy)/self.fy), uncertainty)
            covar_cam_rf_flattened_var_z = torch.transpose(covar_cam_rf.reshape(-1,covar_cam_rf.size()[-1]), 0, 1) # (1,HxW)
            covar_cam_rf_flattened_var_xx = torch.transpose(covar_cam_rf_xx.reshape(-1,covar_cam_rf_xx.size()[-1]), 0, 1) # (1,HxW)
            covar_cam_rf_flattened_var_yy = torch.transpose(covar_cam_rf_yy.reshape(-1,covar_cam_rf_yy.size()[-1]), 0, 1) # (1,HxW)
            covar_cam_rf_flattened_var_xz = torch.transpose(covar_cam_rf_xz.reshape(-1,covar_cam_rf_xz.size()[-1]), 0, 1) # (1,HxW)
            covar_cam_rf_flattened_var_yz = torch.transpose(covar_cam_rf_yz.reshape(-1,covar_cam_rf_yz.size()[-1]), 0, 1) # (1,HxW)
            covar_cam_rf_flattened_var_xy = torch.transpose(covar_cam_rf_xy.reshape(-1,covar_cam_rf_xy.size()[-1]), 0, 1) # (1,HxW)
            covar_cam_rf_flattened = torch.zeros((6,self.num_of_pixels), device=self.cuda)
            covar_cam_rf_flattened[2,:] = covar_cam_rf_flattened_var_z
            covar_cam_rf_flattened[0,:] = covar_cam_rf_flattened_var_xx
            covar_cam_rf_flattened[1,:] = covar_cam_rf_flattened_var_yy
            covar_cam_rf_flattened[3,:] = covar_cam_rf_flattened_var_xy
            covar_cam_rf_flattened[4,:] = covar_cam_rf_flattened_var_xz
            covar_cam_rf_flattened[5,:] = covar_cam_rf_flattened_var_yz
            covar_not_seen = covar_cam_rf_flattened[:,masked_pixels_out_frame] # (3, HxW)
        else:
            covar_not_seen = torch.zeros((6,pt_not_seen.size()[1]), device=self.cuda)
        if self.first_frame:
            self.counter_cloud = torch.ones((1,pt_not_seen.size()[1]), device=self.cuda)
            pos_cloud_cam_rf_updated = pt_not_seen
            covar_cloud_cam_rf_updated = covar_not_seen
            # calculate frame uncertainty, projected prediction on frame, error on frame
            frame_uncertainty = self.calculate_masked_uncertainty(pos_cloud_cam_rf_updated, covar_cloud_cam_rf_updated,mask_in_frame,masked_pixels_in_frame, vj, uj, uncertainty)
        else:
            # for points that have been seen before, update that point in point cloud
            # update mean, variance of prediction and ground-truth 
            valid_pos_cloud = torch.clone(pos_cloud_cam_rf[:,valid_point_indices])
            valid_counter_cloud = torch.clone(self.counter_cloud[:,valid_point_indices])
            valid_covar_cloud = torch.clone(covar_cloud_cam_rf[:,valid_point_indices])
            if uncertainty is not None:
                pos_cloud_cam_rf_updated, covar_cloud_cam_rf_updated = self.update_point_cloud_optimized(valid_pos_cloud, valid_counter_cloud, valid_covar_cloud, pos_cloud_cam_rf, covar_cloud_cam_rf,valid_point_indices, pt_cam_rf_flattened, valid_idx_img_with_repeats, covar_cam_rf_flattened=covar_cam_rf_flattened)
            else:
                pos_cloud_cam_rf_updated, covar_cloud_cam_rf_updated = self.update_point_cloud_optimized(valid_pos_cloud, valid_counter_cloud, valid_covar_cloud, pos_cloud_cam_rf, covar_cloud_cam_rf, valid_point_indices, pt_cam_rf_flattened, valid_idx_img_with_repeats)
            frame_uncertainty = self.calculate_masked_uncertainty(pos_cloud_cam_rf_updated, covar_cloud_cam_rf_updated,mask_in_frame,masked_pixels_in_frame, vj, uj, uncertainty)
            # add new points to cloud
            pos_cloud_cam_rf_updated = torch.cat((pos_cloud_cam_rf_updated,pt_not_seen),axis=1)
            self.counter_cloud = torch.cat((self.counter_cloud,torch.ones((1,pt_not_seen.size()[1]),device=self.cuda)),axis=1)
            covar_cloud_cam_rf_updated = torch.cat((covar_cloud_cam_rf_updated,covar_not_seen),axis=1)
        return frame_uncertainty, pos_cloud_cam_rf_updated, covar_cloud_cam_rf_updated, num_new_points_added
    '''
    Name: update_point_cloud_optimized
    Description: add current image, updating points in point cloud that we are seeing again and adding points to point cloud we are seeing for the first time
    Inputs:
        valid_pos_cloud             : position point cloud of only points that are being seen again in current camera reference frame
        valid_counter_cloud         : counter point cloud of only points that are being seen again in current camera reference frame
        valid_covar_cloud           : covariance point cloud of only points being seen again in current camera reference frame
        pos_cloud_cam_rf            : position point cloud in current camera reference frame
        covar_cloud_cam_rf          : covariance point cloud in current camera reference frame
        valid_point_indices         : indices of mask_in_frame that are nonzero
        pt_cam_rf_flattened         : current image depth prediction flattened tensor 
        valid_idx_img_with_repeats  : indices of image that are nonzero with repeats
        covar_cam_rf_flattened      : current image aleatoric uncertainty prediction flattened tensor 
    Output:
        pos_cloud_cam_rf            : updated position point cloud in current camera reference frame
        covar_cloud_cam_rf          : updated covariance point cloud in current camera reference frame    
    '''
    def update_point_cloud_optimized(self, valid_pos_cloud, valid_counter_cloud, valid_covar_cloud, pos_cloud_cam_rf, covar_cloud_cam_rf, valid_point_indices, pt_cam_rf_flattened, valid_idx_img_with_repeats, covar_cam_rf_flattened=None):
        # calculate number of points updating
        num_pts_updating = valid_pos_cloud.size()[1]
        # update counter
        valid_counter_cloud = valid_counter_cloud+1.0
        # calculate constant 
        M_minus_1_over_M = ((valid_counter_cloud-1.0)/valid_counter_cloud)
        # calculate valid new points in new image 
        pt_cam_rf_valid = pt_cam_rf_flattened[:,valid_idx_img_with_repeats]
        # update iterative mean and iterative variance for prediction
        mean_old = torch.clone(valid_pos_cloud)
        valid_pos_cloud = valid_pos_cloud +(pt_cam_rf_valid-valid_pos_cloud)/valid_counter_cloud
        if covar_cam_rf_flattened is not None:
            valid_covar_img = covar_cam_rf_flattened[:,valid_idx_img_with_repeats]
            # update compressed covariance (6 variables stored)
            partial_covariance_1 = M_minus_1_over_M * valid_covar_cloud # (M-1)/M*cov_old
            # reshape mean vectors and new point vectors from (3,HxW) to (3,1,HxW)
            mean_old_reshaped = torch.reshape(mean_old, (mean_old.size()[0],1,mean_old.size()[1])) # (3,1,HxW)
            mean_new_reshaped = torch.reshape(valid_pos_cloud, (mean_old.size()[0],1,mean_old.size()[1])) # (3,1,HxW)
            new_point_reshaped = torch.reshape(pt_cam_rf_valid, \
                (pt_cam_rf_valid.size()[0],1, \
                pt_cam_rf_valid.size()[1])) # (3,1,HxW)
            # reshape compressed covariance into matrix 
            uncompressed_covar_valid_img = self.uncompress_compressed_symmetric_matrix(a_diag=valid_covar_img[0,:], \
                b=valid_covar_img[3,:], c = valid_covar_img[4,:], d_diag=valid_covar_img[1,:], \
                e = valid_covar_img[5,:], f_diag = valid_covar_img[2,:], num_of_points=valid_covar_img.shape[1]) # (3,3, HxW)
            # calculate matrix multiplications
            valid_counter_cloud = valid_counter_cloud.permute(0,2,1)
            partial_covariance_2_1 = (((valid_counter_cloud-1.0)/valid_counter_cloud) * self.vec_squared(mean_old_reshaped)) - self.vec_squared(mean_new_reshaped) # mean_old_reshaped@torch.transpose(mean_old_reshaped, (1,0,2)) - mean_new_reshaped@torch.transpose(mean_new_reshaped, (1,0,2))
            partial_covariance_2_2 = 1./valid_counter_cloud*(uncompressed_covar_valid_img+self.vec_squared(new_point_reshaped))
            partial_covariance_2 = partial_covariance_2_1 + partial_covariance_2_2
            partial_covariance_2_vec = self.compress_symmetric_matrix(partial_covariance_2, num_of_points=num_pts_updating)
            ## add partial covariance terms up
            valid_covar_cloud = torch.squeeze(partial_covariance_1) + partial_covariance_2_vec
            # for determistic case, numerical errors for very small negative values; only replacing those with 0 
            numerical_errors = torch.logical_and((valid_covar_cloud < 0), (valid_covar_cloud > -1e5))
            valid_covar_cloud[numerical_errors] = 0 # replace numerical error with 0
        else:
            valid_covar_cloud = valid_covar_cloud +torch.mul((pt_cam_rf_valid-mean_old),(pt_cam_rf_valid-valid_pos_cloud))
        pos_cloud_cam_rf[:,valid_point_indices] = valid_pos_cloud
        self.counter_cloud[:,valid_point_indices] = valid_counter_cloud.permute(0,2,1)
        covar_cloud_cam_rf[:,valid_point_indices] = valid_covar_cloud.unsqueeze(2)
        return pos_cloud_cam_rf, covar_cloud_cam_rf
    '''
    Name: vec_squared
    Description: compute x^Tx efficiently using einsum 
    Inputs:
        vec         : vector to be squared 
    Ouputs:
        squared_vec : squared vector
    '''
    def vec_squared(self, vec):
        # k = vec.shape[2]
        # n = 3
        # m = 1
        # j = 3
        squared_vec = torch.einsum('nmk,mjk->njk',vec,vec.permute(1,0,2))
        return squared_vec
    
    '''
    Name: compress_symmetric_matrix
    Description: compress symmetric tensor (3x3xN 9N entries --> 6x1xN 6N entries)  
    Inputs:
        sym_mat         : symmetric tensor to be compressed 
        num_of_points   : third dimension of tensor (number of points)
    Ouputs
        compressed_vec  : compressed tensor (6x1xN)
    '''
    def compress_symmetric_matrix(self, sym_mat, num_of_points):
        sym_mat_num_of_points = num_of_points
        a_diag = torch.reshape(sym_mat[0,0,:], (1, sym_mat_num_of_points))
        b = torch.reshape(sym_mat[0,1,:], (1, sym_mat_num_of_points))
        c = torch.reshape(sym_mat[0,2,:], (1, sym_mat_num_of_points))
        d_diag = torch.reshape(sym_mat[1,1,:], (1, sym_mat_num_of_points))
        e = torch.reshape(sym_mat[1,2,:], (1, sym_mat_num_of_points))
        f_diag = torch.reshape(sym_mat[2,2,:], (1, sym_mat_num_of_points))
        compressed_vec = torch.cat((a_diag, d_diag, f_diag, b, c, e), axis=0)
        return compressed_vec
    '''
    Name: uncompress_compresssed_symmetric_matrix
    Description: uncompress compressed symmetric tensor (6x1xN 6N entries --> 3x3xN 9N entries)
    Inputs:
        a_diag          : (0,0) location vector (along third axis)
        b               : (0,1) location vector (along third axis)
        c               : (0,0) location vector (along third axis)
        d_diag          : (0,1) location vector (along third axis)
        e               : (0,0) location vector (along third axis)
        f_diag          : (0,1) location vector (along third axis)
        num_of_points   : third dimension of tensor (number of points)
    Ouputs
        sym_mat         : uncompressed symmetric tensor
    '''
    def uncompress_compressed_symmetric_matrix(self, a_diag, b, c, d_diag, e, f_diag, num_of_points):
        sym_mat = torch.zeros((3,3,num_of_points), device=self.cuda) # initialize covariance 
        sym_mat[0,0,:] = torch.flatten(a_diag)
        sym_mat[0,1,:] = torch.flatten(b)
        sym_mat[0,2,:] = torch.flatten(c)
        sym_mat[1,1,:] = torch.flatten(d_diag)
        sym_mat[1,2,:] = torch.flatten(e)
        sym_mat[2,2,:] = torch.flatten(f_diag)
        sym_mat[1,0,:] = sym_mat[0,1,:]  # symmetric
        sym_mat[2,0,:] = sym_mat[0,2,:]  # symmetric 
        sym_mat[2,1,:] = sym_mat[1,2,:]  # symmetric 
        return sym_mat

    '''
    Name: uncompress_compresssed_symmetric_matrix_flattened
    Description: uncompress compressed symmetric tensor (6x1xN 6N entries --> 3x(3xN) 9N entries)
    Inputs:
        a_diag          : (0,0) location vector (along third axis)
        b               : (0,1) location vector (along third axis)
        c               : (0,0) location vector (along third axis)
        d_diag          : (0,1) location vector (along third axis)
        e               : (0,0) location vector (along third axis)
        f_diag          : (0,1) location vector (along third axis)
        num_of_points   : third dimension of tensor (number of points)
    Ouputs
        sym_mat         : uncompressed symmetric tensor (3x(3xN))
    '''
    def uncompress_compressed_symmetric_matrix_flattened(self, a_diag, b, c, d_diag, e, f_diag, num_of_points):
        num_cols = 3*num_of_points
        sym_mat = torch.zeros((3,num_cols), device=self.cuda) # initialize covariance 
        sym_mat[0,0:num_cols:3] = a_diag
        sym_mat[0,1:num_cols:3] = b
        sym_mat[0,2:num_cols:3] = c
        sym_mat[1,1:num_cols:3] = d_diag
        sym_mat[1,2:num_cols:3] = e
        sym_mat[2,2:num_cols:3] = f_diag
        sym_mat[1,0:num_cols:3] = sym_mat[0,1:num_cols:3]  # symmetric
        sym_mat[2,0:num_cols:3] = sym_mat[0,2:num_cols:3]  # symmetric 
        sym_mat[2,1:num_cols:3] = sym_mat[1,2:num_cols:3]  # symmetric 
        return sym_mat
    '''
    Name: uncompress_compresssed_symmetric_matrix_flattened_vert
    Description: uncompress compressed symmetric tensor (6x1xN 6N entries --> (3xN)x3 9N entries)
    Inputs:
        a_diag          : (0,0) location vector (along third axis)
        b               : (0,1) location vector (along third axis)
        c               : (0,0) location vector (along third axis)
        d_diag          : (0,1) location vector (along third axis)
        e               : (0,0) location vector (along third axis)
        f_diag          : (0,1) location vector (along third axis)
        num_of_points   : third dimension of tensor (number of points)
    Ouputs
        sym_mat         : uncompressed symmetric tensor ((3xN)x3)
    '''
    def uncompress_compressed_matrix_flattened_vert(self, a_diag, b, c, d_diag, e, f_diag, g, h, i, num_of_points):
        num_rows = 3*num_of_points
        sym_mat = torch.zeros((num_rows,3), device=self.cuda) # initialize covariance 
        sym_mat[0:num_rows:3,0] = a_diag
        sym_mat[0:num_rows:3,1] = b
        sym_mat[0:num_rows:3,2] = c
        sym_mat[1:num_rows:3,1] = d_diag
        sym_mat[1:num_rows:3,2] = e
        sym_mat[2:num_rows:3,2] = f_diag
        sym_mat[1:num_rows:3,0] = g  
        sym_mat[2:num_rows:3,0] = h   
        sym_mat[2:num_rows:3,1] = i   
        return sym_mat

    '''
    Name: compress_vertical_array
    Description: compress symmetric tensor flattened vertically ((3xN)x3 9N entries --> 6x1xN 6N entries 9N entries)
    Inputs:
        a_diag          : (0,0) location vector (along third axis)
        b               : (0,1) location vector (along third axis)
        c               : (0,0) location vector (along third axis)
        d_diag          : (0,1) location vector (along third axis)
        e               : (0,0) location vector (along third axis)
        f_diag          : (0,1) location vector (along third axis)
        num_of_points   : third dimension of tensor (number of points)
    Ouputs
        sym_mat         : uncompressed symmetric tensor (3x(3xN))
    '''
    def compress_vertical_array(self, vert_sym_mat, num_of_points):
        num_rows = 3*num_of_points
        a_diag = vert_sym_mat[0:num_rows:3,0]
        b = vert_sym_mat[0:num_rows:3,1]
        c = vert_sym_mat[0:num_rows:3,2]
        d_diag = vert_sym_mat[1:num_rows:3,1]
        e = vert_sym_mat[1:num_rows:3,2]
        f_diag = vert_sym_mat[2:num_rows:3,2]
        compressed_array = torch.cat((torch.reshape(a_diag,(1,num_of_points)), torch.reshape(d_diag,(1,num_of_points)), 
                                           torch.reshape(f_diag,(1,num_of_points)), torch.reshape(b,(1,num_of_points)),
                                           torch.reshape(c,(1,num_of_points)), torch.reshape(e,(1,num_of_points))), axis=0)
        return compressed_array
    
    '''
    Name: stack_horizontal_array_by_entry_not_symmetric
    Description: reshape horizontal array to be stacked by each entry of 3x(3xN) matrix --> 9xN
    Inputs:
        hor_sym_mat     : uncompressed tensor (3x(3xN))
        num_of_points   : number of covariances/points 
    Ouputs
        stacked_array   : stacked tensor 9xN
    ''' 
    def stack_horizontal_array_by_entry_not_symmetric(self, hor_sym_mat, num_of_points):
        num_cols = 3*num_of_points
        a_diag = hor_sym_mat[0,0:num_cols:3]
        b = hor_sym_mat[0,1:num_cols:3]
        c = hor_sym_mat[0,2:num_cols:3]
        d_diag = hor_sym_mat[1,1:num_cols:3]
        e = hor_sym_mat[1,2:num_cols:3]
        f_diag = hor_sym_mat[2,2:num_cols:3]
        # g (1,0), h (2,0), i (2,1) -- used same lettering as symmetric, compressed case to keep consistent 
        g = hor_sym_mat[1,0:num_cols:3]
        h = hor_sym_mat[2,0:num_cols:3]
        i = hor_sym_mat[2,1:num_cols:3]
        stacked_array = torch.cat((torch.reshape(a_diag,(1,num_of_points)), torch.reshape(d_diag,(1,num_of_points)), 
                                           torch.reshape(f_diag,(1,num_of_points)), torch.reshape(b,(1,num_of_points)),
                                           torch.reshape(c,(1,num_of_points)), torch.reshape(e,(1,num_of_points)), 
                                           torch.reshape(g,(1,num_of_points)), torch.reshape(h,(1,num_of_points)), torch.reshape(i,(1,num_of_points))),axis=0)
        return stacked_array
    
    '''
    Name: calculate_masked_uncertainty
    Description: update frame uncertainty; initialized with aleatoric uncertainty prediction for all pixels, and for pixels that are additional views of points
                 we have seen before, update those pixels with the z component of the updated UfM uncertainty calculated
    Inputs:
        pos_cloud_cam_rf_updated    : updated position point cloud in current camera reference frame
        covar_cloud_cam_rf_updated  : updated covariance point cloud in current camera reference frame    
        mask_in_frame               : mask on point cloud array for which points in point cloud are in frame 
        masked_pixels_in_frame      : mask on frame for which pixels of the frame are in the point cloud 
        uj                          : u pixel locations for each point in point cloud
        vj                          : v pixel locations for each point in point cloud
        aleatoric_unc               : predicted aleatoric uncertainty from DNN
    Output:
        uncertainty                 : UfM uncertainty 
    '''
    def calculate_masked_uncertainty(self,pos_cloud_cam_rf_updated, covar_cloud_cam_rf_updated,mask_in_frame,masked_pixels_in_frame, vj, uj, aleatoric_unc=None):
        if aleatoric_unc is not None: 
            uncertainty = torch.squeeze(torch.clone(aleatoric_unc)) # cloned so we don't rewrite aleatoric uncertainty to be same as multiview uncertainty
        else: 
            uncertainty = torch.zeros((self.height, self.width),device=self.cuda)
        if not self.first_frame:
            vj_valid = vj[mask_in_frame]
            uj_valid = uj[mask_in_frame]
            uncertainty_single_valid = covar_cloud_cam_rf_updated[2,mask_in_frame]
            tic_np = time.perf_counter()
            uncertainty = uncertainty.data.cpu().data.numpy() 
            uncertainty[vj_valid.data.cpu().data.numpy(), uj_valid.data.cpu().data.numpy()] = uncertainty_single_valid.data.cpu().data.numpy()
            uncertainty = torch.from_numpy(uncertainty).to(self.cuda)
            torch.cuda.synchronize(device=self.cuda)
            toc_np = time.perf_counter()
            time_np_conversion = toc_np - tic_np
        assert (torch.min(uncertainty) >= 0),"Problem, predicting negative variance!"
        return uncertainty
    '''
    Name: delete_points_from_point_cloud
    Description: threshold points from point cloud, first throwing away points that are out of view; if still over threshold, then throw away oldest points
    Inputs:
        valid_point_indices         : indices of mask_in_frame that are nonzero
        num_new_points_added        : number of points seen for the first time added to point cloud
    Output:
    '''
    def delete_points_from_point_cloud(self, valid_point_indices, invalid_point_indices, num_new_points_added):
        ufm_cloud_thresholding_start_time = time.time()
        number_pts_deleted_out_of_view = 0
        number_pts_deleted_oldest = 0
        # compute number of points to remove if can keep some out of view ones 
        number_pts_need_to_delete = self.pos_cloud.size()[1] - self.max_cloud_size
        if number_pts_need_to_delete > 0: # over max cloud size 
            if invalid_point_indices.nelement() != 0: # if there exists points outside of this view 
                # if deleting all points out of view would reduce more than we need, only delete oldest points out of view 
                if int(self.pos_cloud.size()[1] - invalid_point_indices.size()[0]) < self.max_cloud_size: 
                    number_invalid_points_we_can_keep = invalid_point_indices.size()[0] - number_pts_need_to_delete
                    invalid_point_indices_to_remove = invalid_point_indices[0:-number_invalid_points_we_can_keep]
                else: 
                    invalid_point_indices_to_remove = invalid_point_indices 
                torch.cuda.synchronize()
                invalid_point_indices_to_remove = torch.squeeze(invalid_point_indices_to_remove, dim = 1)
                mask_keep = torch.ones((self.pos_cloud.size()[1]), device=self.cuda).bool()
                mask_keep[invalid_point_indices_to_remove] = 0
                mask_keep_idx = torch.nonzero(mask_keep)
                self.pos_cloud = torch.squeeze(self.pos_cloud[:,mask_keep_idx])
                self.counter_cloud = torch.squeeze(self.counter_cloud[:,mask_keep_idx], dim = 2)
                self.covar_cloud = torch.squeeze(self.covar_cloud[:,mask_keep_idx])
                number_pts_deleted_out_of_view = min(number_pts_need_to_delete, invalid_point_indices.nelement()) # only set if we actually deleted it (if there existed invalid points outside of this view) 
        # reduce point cloud by oldest points added to point cloud to stay under max cloud size
        if self.pos_cloud.size()[1] > self.max_cloud_size:
            number_pts_deleted_oldest = (self.pos_cloud.size()[1] - self.max_cloud_size)
            self.pos_cloud = self.pos_cloud[:,number_pts_deleted_oldest:]
            self.counter_cloud = self.counter_cloud[:,number_pts_deleted_oldest:]
            self.covar_cloud = self.covar_cloud[:,number_pts_deleted_oldest:]
        torch.cuda.synchronize()
        self.sum_deletion_time += time.time() - ufm_cloud_thresholding_start_time
        return number_pts_deleted_out_of_view, number_pts_deleted_oldest