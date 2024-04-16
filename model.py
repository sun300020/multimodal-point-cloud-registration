#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from pointnet2 import pointnet2_utils
import torch.nn.functional as F
from torch.autograd import Variable
from util import transform_point_cloud
from chamfer_loss import *
from utils import pairwise_distance_batch, PointNet, Pointer, get_knn_index, Discriminator, feature_extractor, img_feature_extractor, feature_fusion, compute_rigid_transformation, get_keypoints
from pointnet_util import sample_and_group_without_normals, square_distance, angle_difference
from common.math import se3
from common.math_torch import se3
class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.num_keypoints = args.n_keypoints
        self.weight_function = Discriminator(args)
        self.nn_margin = args.nn_margin
        
    def forward(self, *input):
        """
            Args:
                src: Source point clouds. Size (B, 3, N)
                tgt: target point clouds. Size (B, 3, M)
                src_embedding: Features of source point clouds. Size (B, C, N)
                tgt_embedding: Features of target point clouds. Size (B, C, M)
                src_idx: Nearest neighbor indices. Size [B * N * k]
                k: Number of nearest neighbors.
                src_knn: Coordinates of nearest neighbors. Size [B, N, K, 3]
                i: i-th iteration.
                tgt_knn: Coordinates of nearest neighbors. Size [B, M, K, 3]
                src_idx1: Nearest neighbor indices. Size [B * N * k]
                idx2:  Nearest neighbor indices. Size [B, M, k]
                k1: Number of nearest neighbors.
            Returns:
                R/t: rigid transformation.
                src_keypoints, tgt_keypoints: Selected keypoints of source and target point clouds. Size (B, 3, num_keypoint)
                src_keypoints_knn, tgt_keypoints_knn: KNN of keypoints. Size [b, 3, num_kepoints, k]
                loss_scl: Spatial Consistency loss.
        """
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        src_idx = input[4]
        k = input[5]
        src_knn = input[6] # [b, n, k, 3]
        i = input[7]
        tgt_knn = input[8] # [b, n, k, 3]
        src_idx1 = input[9] # [b * n * k1]
        idx2 = input[10] #[b, m, k1]
        k1 = input[11]

        batch_size, num_dims_src, num_points = src.size()
        batch_size, _, num_points_tgt = tgt.size()
        batch_size, _, num_points = src_embedding.size()

        ########################## Matching Map Refinement Module ##########################
        distance_map = pairwise_distance_batch(src_embedding, tgt_embedding) #[b, n, m]
        # point-wise matching map
        scores = torch.softmax(-distance_map, dim=2) #[b, n, m]  Eq. (1)
        
        # neighborhood-wise matching map
        src_knn_scores = scores.view(batch_size * num_points, -1)[src_idx1, :]
        src_knn_scores = src_knn_scores.view(batch_size, num_points, k1, num_points) # [b, n, k, m]
        src_knn_scores = pointnet2_utils.gather_operation(src_knn_scores.view(batch_size * num_points, k1, num_points),\
            idx2.view(batch_size, 1, num_points * k1).repeat(1, num_points, 1).view(batch_size * num_points, num_points * k1).int()).view(batch_size,\
                num_points, k1, num_points, k1)[:, :, 1:, :, 1:].sum(-1).sum(2) / (k1-1) # Eq. (2)

        src_knn_scores = self.nn_margin - src_knn_scores
        refined_distance_map = torch.exp(src_knn_scores) * distance_map
        refined_matching_map = torch.softmax(-refined_distance_map, dim=2) # [b, n, m] Eq. (3)

        # pseudo correspondences of source point clouds (pseudo target point clouds)
        src_corr = torch.matmul(tgt, refined_matching_map.transpose(2, 1).contiguous())# [b,3,n] Eq. (4)

        ############################## Inlier Evaluation Module ##############################
        # neighborhoods of pseudo target point clouds
        src_knn_corr = src_corr.transpose(2,1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_knn_corr = src_knn_corr.view(batch_size, num_points, k, num_dims_src)#[b, n, k, 3]

        # edge features of the pseudo target neighborhoods and the source neighborhoods 
        knn_distance = src_corr.transpose(2,1).contiguous().unsqueeze(2) - src_knn_corr #[b, n, k, 3]
        src_knn_distance = src.transpose(2,1).contiguous().unsqueeze(2) - src_knn #[b, n, k, 3]
        
        # inlier confidence
        weight = self.weight_function(knn_distance, src_knn_distance)#[b, 1, n] # Eq. (7)

        # compute rigid transformation 
        R, t = compute_rigid_transformation(src, src_corr, weight) # weighted SVD

        ########################### Preparation for the Loss Function #########################
        # choose k keypoints with highest weights
        src_topk_idx, src_keypoints, tgt_keypoints = get_keypoints(src, src_corr, weight, self.num_keypoints)

        # spatial consistency loss 
        idx_tgt_corr = torch.argmax(refined_matching_map, dim=-1).int() # [b, n]
        identity = torch.eye(num_points_tgt).cuda().unsqueeze(0).repeat(batch_size, 1, 1) # [b, m, m]
        one_hot_number = pointnet2_utils.gather_operation(identity, idx_tgt_corr) # [b, m, n]
        src_keypoints_idx = src_topk_idx.repeat(1, num_points_tgt, 1) # [b, m, num_keypoints]
        keypoints_one_hot = torch.gather(one_hot_number, dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * self.num_keypoints, num_points_tgt)
        #[b, m, num_keypoints] - [b, num_keypoints, m] - [b * num_keypoints, m]
        predicted_keypoints_scores = torch.gather(refined_matching_map.transpose(2, 1), dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * self.num_keypoints, num_points_tgt)
        loss_scl = (-torch.log(predicted_keypoints_scores + 1e-15) * keypoints_one_hot).sum(1).mean()

        # neighorhood information
        src_keypoints_idx2 = src_topk_idx.unsqueeze(-1).repeat(1, 3, 1, k) #[b, 3, num_keypoints, k]
        tgt_keypoints_knn = torch.gather(knn_distance.permute(0,3,1,2), dim = 2, index = src_keypoints_idx2) #[b, 3, num_kepoints, k]

        src_transformed = transform_point_cloud(src, R, t.view(batch_size, 3))
        src_transformed_knn_corr = src_transformed.transpose(2,1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_transformed_knn_corr = src_transformed_knn_corr.view(batch_size, num_points, k, num_dims_src) #[b, n, k, 3]

        knn_distance2 = src_transformed.transpose(2,1).contiguous().unsqueeze(2) - src_transformed_knn_corr #[b, n, k, 3]
        src_keypoints_knn = torch.gather(knn_distance2.permute(0,3,1,2), dim = 2, index = src_keypoints_idx2) #[b, 3, num_kepoints, k]
        return R, t.view(batch_size, 3), src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, loss_scl

class LossFunction(nn.Module):
    def __init__(self, args):
        super(LossFunction, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='sum')
        self.GAL = GlobalAlignLoss()
        self.margin = args.loss_margin

    def forward(self, *input):
        """
            Compute global alignment loss and neighorhood consensus loss
            Args:
                src_keypoints: Keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
            Returns:
                neighborhood_consensus_loss
                global_alignment_loss
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        batch_size = src_keypoints.size()[0]

        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin) 
        emd = self.criterion2(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1))
        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = self.criterion(transformed_srckps_forward, tgt_keypoints)
        knn_consensus_loss = self.criterion(src_keypoints_knn, tgt_keypoints_knn)
        neighborhood_consensus_loss = knn_consensus_loss/k + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss, emd

class LossFunction_kitti(nn.Module):
    def __init__(self, args):
        super(LossFunction_kitti, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='none')
        self.GAL = GlobalAlignLoss()
        self.margin = args.loss_margin

    def forward(self, *input):
        """
            Compute global alignment loss and neighorhood consensus loss
            Args:
                src_keypoints: Selected keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Selected keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
            Returns:
                neighborhood_consensus_loss
                global_alignment_loss
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin) 
        
        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = self.criterion(transformed_srckps_forward, tgt_keypoints).sum(1).sum(1).mean()
        knn_consensus_loss = self.criterion(src_keypoints_knn, tgt_keypoints_knn).sum(1).sum(1).mean()
        neighborhood_consensus_loss = knn_consensus_loss + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss

class RIENET(nn.Module):
    def __init__(self, args):
        super(RIENET, self).__init__()
        self.emb_nn = feature_extractor(args=args)
        self.single_point_embed = PointNet()
        self.forwards = SVDHead(args=args)
        self.iter = args.n_iters 
        if args.dataset == 'kitti':# or args.dataset == 'icl_nuim':
            self.loss = LossFunction_kitti(args)
        else:
            self.loss = LossFunction(args)
        self.criterion2 = ChamferLoss()
        self.k1 = args.k1
        self.k2 = args.k2
        #self.save_path = args.save_path

    def forward(self, *input):
        """ 
            feature extraction.
            Args:
                src = input[0]: Source point clouds. Size [B, 3, N]
                tgt = input[1]: Target point clouds. Size [B, 3, N]
            Returns:
                rotation_ab_pred: Size [B, 3, 3]
                translation_ab_pred: Size [B, 3]
                global_alignment_loss
                consensus_loss
                spatial_consistency_loss
        """

        src = input[0]
        tgt = input[1]
        whole_src = input[2]
        whole_tgt = input[3]

        source_pc_np = whole_src.transpose(1, 2).cpu().numpy()
        target_pc_np = whole_tgt.transpose(1, 2).cpu().numpy()
        '''
        file_path0 = "/root/autodl-tmp/RIENet-main/visual/visualpc0.txt"
        np_src = src[0].detach().cpu().numpy()
        np.savetxt(file_path0, np_src.reshape(np_src.shape[-1], -1), delimiter=" ", fmt="%.6f")
        file_path1 = "/root/autodl-tmp/RIENet-main/visual/visualpc1.txt"
        np_tgt = tgt[0].detach().cpu().numpy()
        np.savetxt(file_path1, np_tgt.reshape(np_tgt.shape[-1], -1), delimiter=" ", fmt="%.6f")
        '''
        


        batch_size, _, _ = src.size()
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        global_alignment_loss, consensus_loss, spatial_consistency_loss = 0.0, 0.0, 0.0

        xyzs1_iter = whole_src
        emd = 0
        for i in range(self.iter):
            src_embedding, src_idx, src_knn, _ = self.emb_nn(src, self.k1)
            tgt_embedding, _, tgt_knn, _ = self.emb_nn(tgt, self.k1)

            src_idx1, _ = get_knn_index(src, self.k2)
            _, tgt_idx = get_knn_index(tgt, self.k2)

            rotation_ab_pred_i, translation_ab_pred_i, src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, spatial_consistency_loss_i\
                = self.forwards(src, tgt, src_embedding, tgt_embedding, src_idx, self.k1, src_knn, i, tgt_knn,\
                    src_idx1, tgt_idx, self.k2)

            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i

            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
            xyzs1_iter = transform_point_cloud(xyzs1_iter, rotation_ab_pred_i, translation_ab_pred_i)
            #estimated_pc_np = xyzs1_iter.transpose(1,2).cpu().detach().numpy()

            neighborhood_consensus_loss_i, global_alignment_loss_i, emd_i = self.loss(src_keypoints, tgt_keypoints,\
                    rotation_ab_pred_i, translation_ab_pred_i, src_keypoints_knn, tgt_keypoints_knn, self.k2, src, tgt)

            chamfer_distance = self.criterion2(xyzs1_iter.permute(0, 2, 1), whole_tgt.permute(0, 2, 1))
            global_alignment_loss += global_alignment_loss_i
            consensus_loss += neighborhood_consensus_loss_i
            spatial_consistency_loss += spatial_consistency_loss_i
            #if i == self.iter-1:
            emd += chamfer_distance
            '''
            np.savetxt(os.path.join(self.save_path,  '%d_estimate.txt' %i), estimated_pc_np[0])
            if i == 0:
                np.savetxt(os.path.join(self.save_path,  '%d_source.txt' %i), source_pc_np[0])
                np.savetxt(os.path.join(self.save_path,  '%d_target.txt' %i), target_pc_np[0])
            '''

    
        return rotation_ab_pred, translation_ab_pred, emd, global_alignment_loss, consensus_loss, spatial_consistency_loss
    

class multimodal_RIENET(nn.Module):
    def __init__(self, args):
        super(multimodal_RIENET, self).__init__()
        self.emb_nn = feature_extractor(args=args)
        self.img_emb_nn = img_feature_extractor(args=args)
        self.fusion = feature_fusion(args=args)
        self.forwards = SVDHead(args=args)
        self.iter = args.n_iters 
        if args.dataset == 'kitti':# or args.dataset == 'icl_nuim':
            self.loss = LossFunction_kitti(args)
        else:
            self.loss = LossFunction(args)
        self.criterion2 = ChamferLoss()
        self.k1 = args.k1
        self.k2 = args.k2
        #self.save_path = args.save_path

    def forward(self, *input):
        """ 
            feature extraction.
            Args:
                src = input[0]: Source point clouds. Size [B, 3, N]
                tgt = input[1]: Target point clouds. Size [B, 3, N]
            Returns:
                rotation_ab_pred: Size [B, 3, 3]
                translation_ab_pred: Size [B, 3]
                global_alignment_loss
                consensus_loss
                spatial_consistency_loss
        """

        src = input[0]
        tgt = input[1]
        whole_src = input[2]
        whole_tgt = input[3]
        img = input[4]

        source_pc_np = whole_src.transpose(1, 2).cpu().numpy()
        target_pc_np = whole_tgt.transpose(1, 2).cpu().numpy()
        '''
        file_path0 = "/root/autodl-tmp/RIENet-main/visual/visualpc0.txt"
        np_src = src[0].detach().cpu().numpy()
        np.savetxt(file_path0, np_src.reshape(np_src.shape[-1], -1), delimiter=" ", fmt="%.6f")
        file_path1 = "/root/autodl-tmp/RIENet-main/visual/visualpc1.txt"
        np_tgt = tgt[0].detach().cpu().numpy()
        np.savetxt(file_path1, np_tgt.reshape(np_tgt.shape[-1], -1), delimiter=" ", fmt="%.6f")
        '''
        


        batch_size, _, _ = src.size()
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        global_alignment_loss, consensus_loss, spatial_consistency_loss = 0.0, 0.0, 0.0
        img_embedding = self.img_emb_nn(img)
        tgt_embedding, _, tgt_knn, _ = self.emb_nn(tgt, self.k1)
        tgt_fusion_embedding = self.fusion(tgt_embedding, img_embedding)
        xyzs1_iter = whole_src
        emd = 0
        for i in range(self.iter):
            src_embedding, src_idx, src_knn, _ = self.emb_nn(src, self.k1)
            src_fusion_embedding = self.fusion(src_embedding, img_embedding)
            #img_embedding = self.img_emb_nn(img)

            src_idx1, _ = get_knn_index(src, self.k2)
            _, tgt_idx = get_knn_index(tgt, self.k2)

            rotation_ab_pred_i, translation_ab_pred_i, src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, spatial_consistency_loss_i\
                = self.forwards(src, tgt, src_fusion_embedding, tgt_fusion_embedding, src_idx, self.k1, src_knn, i, tgt_knn,\
                    src_idx1, tgt_idx, self.k2)

            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i

            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
            xyzs1_iter = transform_point_cloud(xyzs1_iter, rotation_ab_pred_i, translation_ab_pred_i)
            #estimated_pc_np = xyzs1_iter.transpose(1,2).cpu().detach().numpy()

            neighborhood_consensus_loss_i, global_alignment_loss_i, emd_i = self.loss(src_keypoints, tgt_keypoints,\
                    rotation_ab_pred_i, translation_ab_pred_i, src_keypoints_knn, tgt_keypoints_knn, self.k2, src, tgt)

            global_alignment_loss += global_alignment_loss_i
            consensus_loss += neighborhood_consensus_loss_i
            spatial_consistency_loss += spatial_consistency_loss_i
            chamfer_distance = self.criterion2(xyzs1_iter.permute(0, 2, 1), whole_tgt.permute(0, 2, 1))
            #if i == self.iter-1:
            emd += chamfer_distance
            '''
            np.savetxt(os.path.join(self.save_path,  '%d_estimate.txt' %i), estimated_pc_np[0])
            if i == 0:
                np.savetxt(os.path.join(self.save_path,  '%d_source.txt' %i), source_pc_np[0])
                np.savetxt(os.path.join(self.save_path,  '%d_target.txt' %i), target_pc_np[0])
            '''

    
        return rotation_ab_pred, translation_ab_pred, emd, global_alignment_loss, consensus_loss, spatial_consistency_loss

def _go_icp_find_rigid_transform(p, targets, weights):
    A, B = np.copy(p), np.copy(targets)

    centroid_A = np.average(A, axis=0, weights=weights)
    centroid_B = np.average(B, axis=0, weights=weights)

    A -= centroid_A
    B -= centroid_B

    H = np.dot(A.T, B * weights[:, None])
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = np.dot(-R, centroid_A) + centroid_B

    return R, t

def _icp_find_rigid_transform(p_from, p_target):
    A, B = np.copy(p_from), np.copy(p_target)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A -= centroid_A
    B -= centroid_B

    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)

    t = np.dot(-R, centroid_A) + centroid_B

    return R, t

def _icp_Rt_to_matrix(R, t):
    # matrix M = [R, t; 0, 1]
    Rt = np.concatenate((R, np.expand_dims(t.T, axis=-1)), axis=1)
    a = np.concatenate((np.zeros_like(t), np.ones(1)))
    M = np.concatenate((Rt, np.expand_dims(a, axis=0)), axis=0)
    return M


class ICP:
    """ Estimate a rigid-body transform g such that:
        p0 = g.p1
    """
    def __init__(self, p1, p0, whole_src, whole_tgt):
        """ p0.shape == (N, 3)
            p1.shape == (N, 3)
        """
        self.p0 = p0.transpose(1,0)
        self.p1 = p1.transpose(1,0)
        self.whole_src = whole_src.transpose(1,0)
        self.whole_tgt = whole_tgt.transpose(1,0)
        leafsize = 1000
        self.nearest = KDTree(self.p0, leafsize=leafsize)
        self.g_series = None
        self.chamfer_loss = ChamferLoss()
    def compute(self, max_iter, i, save_dir):
        ftol = 0.5
        dim_k = self.p0.shape[1]
        g = np.eye(dim_k + 1, dtype=self.p0.dtype)
        p = np.copy(self.p1)
        whole_p = np.copy(self.whole_src)
        self.g_series = np.zeros((max_iter + 1, dim_k + 1, dim_k + 1), dtype=g.dtype)
        self.g_series[0, :, :] = g

        itr = -1
        for itr in range(max_iter):
            neighbor_idx = self.nearest.query(p)[1]
            targets = self.p0[neighbor_idx]
            R, t = _icp_find_rigid_transform(p, targets)

            new_p = np.dot(R, p.T).T + t  #p->(C,N)
            new_whole_p = np.dot(R, whole_p.T).T + t
            if np.sum(np.abs(p - new_p)) < ftol:
                break
            

            p = np.copy(new_p)
            whole_p = np.copy(new_whole_p)
            dg = _icp_Rt_to_matrix(R, t)
            new_g = np.dot(dg, g)
            g = np.copy(new_g)
            self.g_series[itr + 1, :, :] = g

        self.g_series[(itr+1):, :, :] = g

        # 将当前变换后的源点云和目标点云传递给 Chamfer Loss 计算
        source_cloud_tensor = torch.tensor(new_whole_p, dtype=torch.float32)   #(N,3)
        source_cloud_tensor = torch.unsqueeze(source_cloud_tensor, dim=0)  #(B,N,3)
        target_cloud_tensor = torch.tensor(self.whole_tgt, dtype=torch.float32)
        target_cloud_tensor = torch.unsqueeze(target_cloud_tensor, dim=0)
        # 保存 source_cloud_tensor 和 target_cloud_tensor
        ori_source_cloud_tensor = torch.tensor(self.whole_src, dtype=torch.float32)   #(N,3)
        np.savetxt(os.path.join(save_dir, f'whole_src_ori_{i}.txt'), ori_source_cloud_tensor.numpy(), delimiter=' ')
        np.savetxt(os.path.join(save_dir, f'whole_src_{i}.txt'), source_cloud_tensor.squeeze().numpy(), delimiter=' ')
        np.savetxt(os.path.join(save_dir, f'whole_tgt_{i}.txt'), target_cloud_tensor.squeeze().numpy(), delimiter=' ')

        # 将当前变换后的源点云和目标点云传递给 Chamfer Loss 计算
        source = torch.tensor(new_p, dtype=torch.float32)   #(N,3)
        source = torch.unsqueeze(source, dim=0)  #(B,N,3)
        target = torch.tensor(self.p0, dtype=torch.float32)
        target = torch.unsqueeze(target, dim=0)
        # 保存 source_cloud_tensor 和 target_cloud_tensor
        ori_source = torch.tensor(self.p1, dtype=torch.float32)   #(N,3)
        np.savetxt(os.path.join(save_dir, f'src_ori_{i}.txt'), ori_source.numpy(), delimiter=' ')
        np.savetxt(os.path.join(save_dir, f'src_{i}.txt'), source.squeeze().numpy(), delimiter=' ')
        np.savetxt(os.path.join(save_dir, f'tgt_{i}.txt'), target.squeeze().numpy(), delimiter=' ')
        chamfer_distance = self.chamfer_loss(source_cloud_tensor, target_cloud_tensor)

        return g, p, (itr + 1), chamfer_distance
    

########Go-ICP#########################################
class GoICP:
    def __init__(self, p1, p0, whole_src, whole_tgt):
        """ p0.shape == (N, 3)
            p1.shape == (N, 3)
        """
        self.p0 = p0.transpose(1, 0)
        self.p1 = p1.transpose(1, 0)
        self.whole_src = whole_src.transpose(1, 0)
        self.whole_tgt = whole_tgt.transpose(1, 0)
        leafsize = 1000
        self.nearest = KDTree(self.p0, leafsize=leafsize)
        self.g_series = None
        self.chamfer_loss = ChamferLoss()

    def calculate_weights(self, p, targets):
        distances = np.linalg.norm(p - targets, axis=1)
        max_distance = np.max(distances)
        weights = np.exp(-distances / max_distance)
        return weights

    def compute(self, max_iter, i, save_dir):
        ftol = 0.5
        dim_k = self.p0.shape[1]
        g = np.eye(dim_k + 1, dtype=self.p0.dtype)
        p = np.copy(self.p1)
        whole_p = np.copy(self.whole_src)
        self.g_series = np.zeros((max_iter + 1, dim_k + 1, dim_k + 1), dtype=g.dtype)
        self.g_series[0, :, :] = g

        itr = -1
        for itr in range(max_iter):
            neighbor_idx = self.nearest.query(p)[1]
            targets = self.p0[neighbor_idx]
            
            # Go-ICP: 使用一般的误差度量和权重
            weights = self.calculate_weights(p, targets)
            R, t = _go_icp_find_rigid_transform(p, targets, weights)
            
            new_p = np.dot(R, p.T).T + t
            new_whole_p = np.dot(R, whole_p.T).T
            if np.sum(np.abs(p - new_p)) < ftol:
                break
            
            p = np.copy(new_p)
            whole_p = np.copy(new_whole_p)
            dg = _icp_Rt_to_matrix(R, t)
            new_g = np.dot(dg, g)
            g = np.copy(new_g)
            self.g_series[itr + 1, :, :] = g

        self.g_series[(itr + 1):, :, :] = g
        # 将当前变换后的源点云和目标点云传递给 Chamfer Loss 计算
        source_cloud_tensor = torch.tensor(new_whole_p, dtype=torch.float32)   #(N,3)
        source_cloud_tensor = torch.unsqueeze(source_cloud_tensor, dim=0)  #(B,N,3)
        target_cloud_tensor = torch.tensor(self.whole_tgt, dtype=torch.float32)
        target_cloud_tensor = torch.unsqueeze(target_cloud_tensor, dim=0)
        # 保存 source_cloud_tensor 和 target_cloud_tensor
        ori_source_cloud_tensor = torch.tensor(self.whole_src, dtype=torch.float32)   #(N,3)
        np.savetxt(os.path.join(save_dir, f'whole_src_ori_{i}.txt'), ori_source_cloud_tensor.numpy(), delimiter=' ')
        np.savetxt(os.path.join(save_dir, f'whole_src_{i}.txt'), source_cloud_tensor.squeeze().numpy(), delimiter=' ')
        np.savetxt(os.path.join(save_dir, f'whole_tgt_{i}.txt'), target_cloud_tensor.squeeze().numpy(), delimiter=' ')

        # 将当前变换后的源点云和目标点云传递给 Chamfer Loss 计算
        source = torch.tensor(new_p, dtype=torch.float32)   #(N,3)
        source = torch.unsqueeze(source, dim=0)  #(B,N,3)
        target = torch.tensor(self.p0, dtype=torch.float32)
        target = torch.unsqueeze(target, dim=0)
        # 保存 source_cloud_tensor 和 target_cloud_tensor
        ori_source = torch.tensor(self.p1, dtype=torch.float32)   #(N,3)
        np.savetxt(os.path.join(save_dir, f'src_ori_{i}.txt'), ori_source.numpy(), delimiter=' ')
        np.savetxt(os.path.join(save_dir, f'src_{i}.txt'), source.squeeze().numpy(), delimiter=' ')
        np.savetxt(os.path.join(save_dir, f'tgt_{i}.txt'), target.squeeze().numpy(), delimiter=' ')
        chamfer_distance = self.chamfer_loss(source_cloud_tensor, target_cloud_tensor)

        return g, p, (itr + 1), chamfer_distance
######## RPM-Net delete###############################
_EPS = 1e-5  # To prevent division by zero
_raw_features_sizes = {'xyz': 3, 'dxyz': 3}
_raw_features_order = {'xyz': 0, 'dxyz': 1}

class ParameterPredictionNet(nn.Module):
    def __init__(self, weights_dim):
        """PointNet based Parameter prediction network

        Args:
            weights_dim: Number of weights to predict (excluding beta), should be something like
                         [3], or [64, 3], for 3 types of features
        """

        super().__init__()

        self.weights_dim = weights_dim

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2 + np.prod(weights_dim)),
        )



    def forward(self, x):
        """ Returns alpha, beta, and gating_weights (if needed)

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """

        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha

def get_prepool(in_dim, out_dim):
    """Shared FC part in PointNet before max pooling"""
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
    )
    return net


def get_postpool(in_dim, out_dim):
    """Linear layers in PointNet after max pooling

    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels. Typically smaller than in_dim

    """
    net = nn.Sequential(
        nn.Conv1d(in_dim, in_dim, 1),
        nn.GroupNorm(8, in_dim),
        nn.ReLU(),
        nn.Conv1d(in_dim, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
        nn.Conv1d(out_dim, out_dim, 1),
    )

    return net

class FeatExtractionEarlyFusion(nn.Module):
    """Feature extraction Module that extracts hybrid features"""
    def __init__(self, features, feature_dim, radius, num_neighbors):
        super(FeatExtractionEarlyFusion, self).__init__()

        self.radius = radius
        self.n_sample = num_neighbors

        self.features = sorted(features, key=lambda f: _raw_features_order[f])

        # Layers
        raw_dim = np.sum([_raw_features_sizes[f] for f in self.features])  # number of channels after concat
        self.prepool = get_prepool(raw_dim, feature_dim * 2)
        self.postpool = get_postpool(feature_dim * 2, feature_dim)

    def forward(self, xyz):
        """Forward pass of the feature extraction network

        Args:
            xyz: (B, N, 3)
            normals: (B, N, 3)

        Returns:
            cluster features (B, N, C)

        """
        features = sample_and_group_without_normals(-1, self.radius, self.n_sample, xyz)
        features['xyz'] = features['xyz'][:, :, None, :]

        # Gate and concat
        concat = []
        for i in range(len(self.features)):
            f = self.features[i]
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)

        # Prepool_FC, pool, postpool-FC
        new_feat = fused_input_feat.permute(0, 3, 2, 1)  # [B, 10, n_sample, N]
        new_feat = self.prepool(new_feat)

        pooled_feat = torch.max(new_feat, 2)[0]  # Max pooling (B, C, N)

        post_feat = self.postpool(pooled_feat)  # Post pooling dense layers
        cluster_feat = post_feat.permute(0, 2, 1)
        cluster_feat = cluster_feat / torch.norm(cluster_feat, dim=-1, keepdim=True)

        return cluster_feat  # (B, N, C)

def match_features(feat_src, feat_ref, metric='l2'):
    """ Compute pairwise distance between features

    Args:
        feat_src: (B, J, C)
        feat_ref: (B, K, C)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
         in the src agrees with every point in the ref.
    """
    assert feat_src.shape[-1] == feat_ref.shape[-1]

    if metric == 'l2':
        dist_matrix = square_distance(feat_src, feat_ref)
    elif metric == 'angle':
        feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
        feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)

        dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
    else:
        raise NotImplementedError

    return dist_matrix

def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    #transform = torch.cat((rot_mat, translation), dim=2)
    return rot_mat, translation

def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError

class RPMNet(nn.Module):
    def __init__(self, args):
        super(RPMNet,self).__init__()
        self.feat_extractor = FeatExtractionEarlyFusion(
            features=args.features, feature_dim=args.feat_dim,
            radius=args.radius, num_neighbors=args.num_neighbors)
        self.weights_net = ParameterPredictionNet(weights_dim=[0])
        self.add_slack = not args.no_slack
        self.num_sk_iter = args.num_sk_iter
        self.iter = args.n_iters 
        self.criterion2 = ChamferLoss()
    def compute_affinity(self, beta, feat_distance, alpha=0.5):
        """Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
        if isinstance(alpha, float):
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
        else:
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
        return hybrid_affinity

    def forward(self, *input):
        """Forward pass for RPMNet

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, 6)
                    'points_ref': Reference points (B, K, 6)
            num_iter (int): Number of iterations. Recommended to be 2 for training

        Returns:
            transform: Transform to apply to source points such that they align to reference
            src_transformed: Transformed source points
        """
        endpoints = {}

        src = input[0]
        tgt = input[1]
        whole_src = input[2]
        whole_tgt = input[3]
        xyz_src_t = src

        transforms = []
        all_gamma, all_perm_matrices, all_weighted_ref = [], [], []
        all_beta, all_alpha = [], []
        for i in range(self.iter):

            beta, alpha = self.weights_net([xyz_src_t.transpose(2,1), tgt.transpose(2,1)])
            feat_src = self.feat_extractor(xyz_src_t.transpose(2,1))
            feat_ref = self.feat_extractor(tgt.transpose(2,1))

            feat_distance = match_features(feat_src, feat_ref)
            affinity = self.compute_affinity(beta, feat_distance, alpha=alpha)

            # Compute weighted coordinates
            log_perm_matrix = sinkhorn(affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
            perm_matrix = torch.exp(log_perm_matrix)
            weighted_ref = perm_matrix @ tgt.transpose(2,1) / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)

            # Compute transform and transform points
            rotation_ab_pred, translation_ab_pred = compute_rigid_transform(src.transpose(2,1), weighted_ref, weights=torch.sum(perm_matrix, dim=2))
            xyz_src_t = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred.squeeze())
            whole_xyz_src_t = transform_point_cloud(whole_src, rotation_ab_pred, translation_ab_pred.squeeze())
            transform = torch.cat((rotation_ab_pred, translation_ab_pred), dim=2)

            transforms.append(transform)
            all_gamma.append(torch.exp(affinity))
            all_perm_matrices.append(perm_matrix)
            all_weighted_ref.append(weighted_ref)
            all_beta.append(to_numpy(beta))
            all_alpha.append(to_numpy(alpha))

        endpoints['perm_matrices_init'] = all_gamma
        endpoints['perm_matrices'] = all_perm_matrices
        endpoints['weighted_ref'] = all_weighted_ref
        endpoints['beta'] = np.stack(all_beta, axis=0)
        endpoints['alpha'] = np.stack(all_alpha, axis=0)
        chamfer_distance = self.criterion2(whole_xyz_src_t.permute(0, 2, 1), whole_tgt.permute(0, 2, 1))
        
        return transforms, endpoints, rotation_ab_pred, translation_ab_pred.squeeze(), chamfer_distance
###################################################