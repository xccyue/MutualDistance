from collections import defaultdict
import os
from typing import Any, Tuple
import time
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.basemodel import MotionGRU, SceneNet, SDFNet, GCN_S, GCN_H, MotionGRU_S1
from utils.utilities import Console, Ploter
from utils.visualization import render_attention, frame2video, render_reconstructed_motion_in_scene, render_sample_k_motion_in_scene
import utils.configuration as config
import trimesh
from utils.model_utils import GeometryTransformer
# from human_body_prior.tools.model_loader import load_model
# from human_body_prior.models.vposer_model import VPoser
from utils.smplx_util import SMPLX_Util, marker_indic
import smplx
from utils.geo_utils import smplx_signed_distance
import pickle
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from utils.data_utils import get_dct_matrix

import numpy as np
import pandas as pd
# import kaolin
# from kaolin.ops.mesh import index_vertices_by_faces

EPOCH_REPORT_TEMPLATE = """
----------------------summary----------------------
[train] train_total_loss: {train_total_loss}
[train] train_rec_loss: {train_rec_loss}
[train] train_rec_trans_loss: {train_rec_trans_loss}
[train] train_rec_orient_loss: {train_rec_orient_loss}
[train] train_rec_body_pose_loss: {train_rec_body_pose_loss}
[train] train_rec_hand_pose_loss: {train_rec_hand_pose_loss}
[train] train_sdf_loss: {train_rec_sdf_loss}
[val] val_total_loss: {val_total_loss}
[val] val_rec_loss: {val_rec_loss}
[val] val_rec_trans_loss: {val_rec_trans_loss}
[val] val_rec_orient_loss: {val_rec_orient_loss}
[val] val_rec_body_pose_loss: {val_rec_body_pose_loss}
[val] val_rec_hand_pose_loss: {val_rec_hand_pose_loss}
[val] val_sdf_loss: {val_rec_sdf_loss}
[val] val_hsdf_loss: {val_rec_hsdf_loss}
[val] MPJPE {val_mpjpe}
[val] MPVPE {val_mpvpe}
[val] pose_error {val_pose_error}
[val] path_error {val_path_error}
"""
EPOCH_VAL_REPORT_TEMPLATE = """
----------------------summary----------------------
[val] val_total_loss: {val_total_loss}
[val] val_rec_loss: {val_rec_loss}
[val] val_rec_trans_loss: {val_rec_trans_loss}
[val] val_rec_orient_loss: {val_rec_orient_loss}
[val] val_rec_body_pose_loss: {val_rec_body_pose_loss}
[val] val_rec_hand_pose_loss: {val_rec_hand_pose_loss}
[val] val_sdf_loss: {val_rec_sdf_loss}
[val] val_hsdf_loss: {val_rec_hsdf_loss}
[val] MPJPE {val_mpjpe}
[val] MPVPE {val_mpvpe}
[val] pose_error {val_pose_error}
[val] path_error {val_path_error}
"""

BEST_REPORT_TEMPLATE = """
----------------------best----------------------
[best] best epoch: {best_epoch}
[best] best_total_loss: {best_total_loss}
[best] best_rec_loss: {best_rec_loss}
[best] best_rec_trans_loss: {best_rec_trans_loss}
[best] best_rec_orient_loss: {best_rec_orient_loss}
[best] best_rec_body_pose_loss: {best_rec_body_pose_loss}
[best] best_rec_hand_pose_loss: {best_rec_hand_pose_loss}
[best] best_sdf_loss: {best_rec_sdf_loss}
"""

def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.5)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


class MotionSolver():
    def __init__(self, conf: Any, dataloader: dict):
        self.config = conf
        self.scene_net_s2 = SceneNet(self.config).to(self.config.device)
        self.sdf_net = SDFNet(self.config).to(self.config.device)
        self.motion_decoder = MotionGRU(self.config).to(self.config.device)

        self.scene_net_s1 = SceneNet(self.config).to(self.config.device)
        self.sdf_gcn = GCN_S(self.config).to(self.config.device)
        self.hsdf_gcn = GCN_H(self.config).to(self.config.device)
        self.motion_encoder = MotionGRU_S1(self.config).to(self.config.device)
        self.dct_n = config.dct_n

        self.dataloader = dataloader
        scene_point_path = config.dataset_scene_points
        with open(scene_point_path,'rb') as f:
            data = pickle.load(f)
        data = torch.FloatTensor(data)


        scene_points = data
        scene_points = torch.tensor(scene_points).cuda().float()
        self.scene_points = scene_points.unsqueeze(0).repeat(config.batch_size*60,1,1)
        self.scene_points1 = scene_points.unsqueeze(0).repeat(config.batch_size*90,1,1)
        self.rootidx = 14

 
        dct_n = 60
        dct_m_in, _ = get_dct_matrix(30 + 60)
        dct_m_out, _ = get_dct_matrix(30 + 60)
        pad_idx = np.repeat([30 - 1], 60)
        i_idx = np.append(np.arange(0,30),pad_idx)

        self.optimizer_h = optim.Adam(
            [
                {'params': list(self.scene_net_s2.parameters())},
                {'params': list(self.motion_decoder.parameters())},
                {'params': list(self.sdf_net.parameters())},
                {'params': list(self.scene_net_s1.parameters())},
                {'params': list(self.sdf_gcn.parameters())},
                {'params': list(self.hsdf_gcn.parameters())},
                {'params': list(self.motion_encoder.parameters())}
                
            ],
            lr = self.config.lr
        )
        # print("model size s1", self.print_model_size(self.scene_net_s1)+self.print_model_size(self.sdf_gcn)+self.print_model_size(self.hsdf_gcn)+self.print_model_size(self.motion_encoder))
        # print("model size s2", self.print_model_size(self.scene_net_s2) + self.print_model_size(self.motion_decoder))

        self.scheduler = get_scheduler(self.optimizer_h, policy='step', decay_step=30)

        # log
        # contains all necessary info for all phases
        self.log = {phase: {} for phase in ["train", "val"]}
        self.dump_keys = ['loss', 'rec_loss',  'rec_trans_loss', 'rec_sdf_loss',  'rec_hsdf_loss']

        self.best = {
            'epoch': 0,
            'loss': float("inf"),
            'rec_loss': float("inf")
        }
    def print_model_size(self, model):
        num_params = sum(p.numel() for p in list(model.parameters())) / 1000000.0
        model_size_bytes = sum(p.element_size() * p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = model_size_bytes / (1024 * 1024)
        print(f"Total parameters M: {num_params}")
        print(f"Total size (MB): {model_size_mb:.2f}")
        return num_params
    def _save_state_dict(self, epoch: int, name: str):

        # saved_cond_net_state_dict = {k: v for k, v in self.cond_net.state_dict().items() if 'bert' not in k} # don't save bert weights
        torch.save({
            'epoch': epoch + 1,
            'scene_net_s2_state_dict': self.scene_net_s2.state_dict(),
            'sdf_net_state_dict': self.sdf_net.state_dict(),
            'motion_decoder_state_dict': self.motion_decoder.state_dict(),
            'optimizer_h_state_dict': self.optimizer_h.state_dict(),
            'scene_net_s1_state_dict': self.scene_net_s1.state_dict(),
            'sdf_gcn_state_dict': self.sdf_gcn.state_dict(),
            'hsdf_gcn_state_dict': self.hsdf_gcn.state_dict(),
            'motion_encoder_state_dict':self.motion_encoder.state_dict()
        }, os.path.join(self.config.log_dir, '{}.pth'.format(name)))

    def _load_state_dict(self):
        if self.config.resume_model != '':
     
            if os.path.isdir(self.config.resume_model):
                ckp_file = os.path.join(self.config.resume_model, 'epoch0.pth')
            elif os.path.isfile(self.config.resume_model):
                ckp_file = self.config.resume_model
            else:
                return 0
            state_dict = torch.load(ckp_file)
            self.motion_decoder.load_state_dict(state_dict['motion_decoder_state_dict'])
            self.sdf_net.load_state_dict(state_dict['sdf_net_state_dict'])
            self.scene_net_s2.load_state_dict(state_dict['scene_net_s2_state_dict'])
            self.motion_encoder.load_state_dict(state_dict['motion_encoder_state_dict'])
            self.sdf_gcn.load_state_dict(state_dict['sdf_gcn_state_dict'])
            self.hsdf_gcn.load_state_dict(state_dict['hsdf_gcn_state_dict'])
            self.scene_net_s1.load_state_dict(state_dict['scene_net_s1_state_dict'])
            # self.optimizer_h.load_state_dict(state_dict=['optimizer_h_state_dict'])
            # self.scheduler.load_state_dict(state_dict=['scheduler_state_dict'])
            Console.log('Load checkpoint: {} start from {}'.format(ckp_file, state_dict['epoch']))
            
        else:
           
            if os.path.isdir(self.config.resume_model_s2):
                ckp_file = os.path.join(self.config.resume_model_s2, 'epoch40.pth')
            elif os.path.isfile(self.config.resume_model_s2):
                ckp_file = self.config.resume_model_s2
            else:
                return 0
        
            print("the ckp file is ", ckp_file)
            state_dict = torch.load(ckp_file)
            print(state_dict.keys())
            self.motion_decoder.load_state_dict(state_dict['motion_decoder_state_dict'])
            self.sdf_net.load_state_dict(state_dict['sdf_net_state_dict'])
            self.scene_net_s2.load_state_dict(state_dict['scene_net_state_dict'])
            Console.log('Load checkpoint: {}. start from {}'.format(ckp_file, state_dict['epoch']))

            if os.path.isdir(self.config.resume_model_s1):
                ckp_file = os.path.join(self.config.resume_model_s1, 'epoch40.pth')
            elif os.path.isfile(self.config.resume_model_s1):
                ckp_file = self.config.resume_model_s1
            else:
                return 0
        
            print("the ckp file is ", ckp_file)
            state_dict = torch.load(ckp_file)
            print(state_dict.keys())
            self.motion_encoder.load_state_dict(state_dict['motion_encoder_state_dict'])
            self.sdf_gcn.load_state_dict(state_dict['sdf_gcn_state_dict'])
            self.hsdf_gcn.load_state_dict(state_dict['hsdf_gcn_state_dict'])
            self.scene_net_s1.load_state_dict(state_dict['scene_net_state_dict'])
            Console.log('Load checkpoint: {}. start from {}'.format(ckp_file,state_dict['epoch']))





        return 0
    def _set_phase(self, phase: str):
        if phase == "train":
            self.sdf_net.train()
            self.scene_net_s2.train()
            # self.motion_net.train()
            self.motion_decoder.train()
            
            self.scene_net_s1.train()
            self.hsdf_gcn.train()
            self.sdf_gcn.train()
            self.motion_encoder.train()
        elif phase == "val":
            self.sdf_net.eval()
            self.scene_net_s2.eval()
            # self.motion_net.train()
            self.motion_decoder.eval()
            
            self.scene_net_s1.eval()
            self.hsdf_gcn.eval()
            self.sdf_gcn.eval()
            self.motion_encoder.eval()
        else:
            raise Exception("Invalid phase")

    def __call__(self):

        start_epoch = self._load_state_dict()
        
        for epoch_id in range(start_epoch, self.config.num_epoch):
            Console.log('epoch {:0>5d} starting...'.format(epoch_id))

            ## train
            self._set_phase('train')
            self._train(self.dataloader['train'], epoch_id)

            ## val
            with torch.no_grad():
                self._set_phase('val')
                self._val(self.dataloader['val'], epoch_id)

            ## report log
            self._epoch_report(epoch_id)
            self._dump_log(epoch_id)

        # print best
        self._best_report()

        # save model
        Console.log("saving last models...\n")
        self._save_state_dict(epoch_id, 'model_last')

    def _forward(
        self, history_body, scene_sdf, sdf_in, hsdf_in
    ):
        """ Forward function to predict

        Args:

            args: smplx parameters group for traning, (trans, orient, pose_body, pose_hand<optional>)
            # trans B S 3
            # orient B S 3
            # pose body B S 63
            # pose hand B S 90
            # scene_sdf B 125 125 125
            # motion sdf B S N

        Return:
            reconstruct results and (mu, logvar) pairs
        """
        B, S,Np, _ = history_body.shape
        # print(trans_shape,orient_shape,pose_body_shape,pose_hand_shape)
        # B S 162
        history_body = history_body.reshape(B,S,Np*3)
        # print("************")
        # B S 162
        # print(history_motion[:,-1,:3])
        history_motion = history_body.permute(1,0,2)

        
        # print(history_motion[-1,:,:3])
        # print("----------------")
        # print("input shape", history_motion.shape) 25 64 162
        scene_sdf = scene_sdf.unsqueeze(1)
        # start_time_s1 = time.time()
        fs = self.scene_net_s1(scene_sdf)
        # fs B 128
        fh = self.motion_encoder(history_motion)
        # fh B 128
        # sdf_in B N L
        # beta B 10
        pred_sdf = self.sdf_gcn(sdf_in,fs,fh)
        pred_hsdf = self.hsdf_gcn(hsdf_in,fs,fh)
        # end_time_s1 = time.time()



        T = 90
        N = sdf_in.shape[1]
        dct_n = self.dct_n
        _, idct_m = get_dct_matrix(T)
        idct_m = torch.FloatTensor(idct_m).cuda()
        B = pred_sdf.shape[0]
        pred_sdf = pred_sdf.view(B,-1)
        # L B*N
        pred_t = pred_sdf.view(-1, dct_n).transpose(0,1)
        pred_expmap = torch.matmul(idct_m[:, :dct_n], pred_t).transpose(0, 1).contiguous().view(-1, N,
                                                                                               T).transpose(1, 2)
        hl = self.config.history_len

        motion_sdf = pred_expmap


        T = 90
        N = hsdf_in.shape[1]
        dct_n = self.dct_n
        _, idct_m = get_dct_matrix(T)
        idct_m = torch.FloatTensor(idct_m).cuda()
        B = pred_hsdf.shape[0]
        pred_hsdf = pred_hsdf.view(B,-1)
        # L B*N
        pred_ht = pred_hsdf.view(-1, dct_n).transpose(0,1)
        pred_hexpmap = torch.matmul(idct_m[:, :dct_n], pred_ht).transpose(0, 1).contiguous().view(-1, N,
                                                                                               T).transpose(1, 2)
        hl = self.config.history_len

        human_sdf = pred_hexpmap






        fs = self.scene_net_s2(scene_sdf)
        fsdf = self.sdf_net(motion_sdf)
        hsdf = human_sdf.permute(1,0,2)
        fsdf = fsdf.permute(1,0,2)
        output = self.motion_decoder(history_motion, fs, fsdf, hsdf)
        output = output.permute(1,0,2)
        output = output.reshape(B,60,Np,3)
        pred_body = output


        return pred_body, motion_sdf, human_sdf



    def get_sdf_grid_batch(self,  verts, scene_sdf1, scene_index1):
        
        B = verts.shape[0]
        S = verts.shape[1]
        B,S,N,_ = verts.shape
        lo1 = verts.reshape(B,S*N,3)
        scene_index1 = scene_index1.unsqueeze(1)
        lo1 =2*((lo1 - scene_index1[:,:,:3])/(scene_index1[:,:,3:] - scene_index1[:,:,:3]) - 0.5)
        sdf1 = F.grid_sample(scene_sdf1[:,None],lo1[:,None,None,:,[2,1,0]], align_corners=True)
        sdf1 = sdf1.reshape(B,S,N)

        return sdf1


    def _get_hsdf(self, verts, scene_points):
        B = verts.shape[0]
        S = verts.shape[1]
        N = verts.shape[2]
        # print('ver', verts.shape, scene_points.shape)
        # print(scene_points.shape)
        verts = verts.reshape(B*S,N,-1)
        # print(verts.shape, scene_point.shape)
        distance = torch.cdist(scene_points,verts)
        distance = torch.min(distance,dim=2)[0]
        distance = distance.reshape(B,S,-1)

        return distance

            
    def _compute_rec_error(self, future_trans, future_orient,future_pose_body,future_pose_hand,
                pred_trans, pred_orient, pred_pose_body, pred_pose_hand, betas,
               scene_radius, future_motion_sdf, motion_transformation):

        B, S, _ = future_trans.shape
        
        ## rec loss, body vertices
        ## 1. get ground truth body vertices
        motion_transformation = motion_transformation.unsqueeze(1).repeat(1,S,1)
        pred_orient = GeometryTransformer.convert_to_3D_rot(pred_orient.reshape(-1, 6)).reshape(B,S,3)
        future_orient = GeometryTransformer.convert_to_3D_rot(future_orient.reshape(-1, 6)).reshape(B,S,3)
        pred_trans = pred_trans+motion_transformation
        future_trans = future_trans + motion_transformation
        verts_gt, joints_gt = self._get_body_vertices(
            future_trans.reshape(B * S, -1),
            future_orient.reshape(B * S, -1),
            betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1),
            future_pose_body.reshape(B * S, -1),
            future_pose_hand.reshape(B * S, -1)
        )
        ## 2. get rec body vertices
        
        verts_rec, joints_rec = self._get_body_vertices(
            pred_trans.reshape(B * S, -1),
            pred_orient.reshape(B * S, -1),
            betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1),
            pred_pose_body.reshape(B * S, -1),
            pred_pose_hand.reshape(B * S, -1),
        )
        ## 3. mpvpe
        verts_gt = verts_gt.reshape(B, S, -1, 3)
        verts_rec = verts_rec.reshape(B, S, -1, 3)
        # rec_vertex_error = (F.l1_loss(
        #     verts_gt, verts_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum()
        rec_vertex_error = torch.sqrt(((verts_gt - verts_rec) ** 2).sum(-1)).mean(-1) # <B, S, N, 3>

        ## 4. mpjpe
        joints_gt = joints_gt.reshape(B, S, -1, 3)
        joints_rec = joints_rec.reshape(B, S, -1, 3)
        # rec_joints_error = (F.l1_loss(
        #     joints_gt, joints_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum()
        rec_joints_error = torch.sqrt(((joints_gt - joints_rec) ** 2).sum(-1)).mean(-1)  # <B, S, N, 3>
        path_error = torch.sqrt(((pred_trans - future_trans) ** 2).sum(-1)).mean(-1)
        # print(path_error.shape,pred_trans.shape,future_trans.shape)
        verts_gt, joints_gt = self._get_body_vertices(
            torch.zeros((B*S,3)).cuda(),
            future_orient.reshape(B * S, -1),
            betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1),
            future_pose_body.reshape(B * S, -1),
            future_pose_hand.reshape(B * S, -1)
        )
        ## 2. get rec body vertices
        
        verts_rec, joints_rec = self._get_body_vertices(
            torch.zeros((B*S,3)).cuda(),
            pred_orient.reshape(B * S, -1),
            betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1),
            pred_pose_body.reshape(B * S, -1),
            pred_pose_hand.reshape(B * S, -1),
        )
        # print(joints_gt.shape,joints_rec.shape)
        pose_error = torch.sqrt(((joints_gt - joints_rec) ** 2).sum(-1)).mean(-1)
        # print(pose_error.shape)



        return  path_error, pose_error
    


    def _cal_loss(self, future_pose, pred_pose, future_hsdf, future_msdf, rootidx, scene_sdf, scene_index, scene_points):
        """ Compute loss, rec_loss is `l1_loss`

        Args:
            x: ground truth, (trans, orient, body_pose)
            rec_x: reconstructed result, (trans, orient, body_pose)
            mu: 
            logvar:
            motion_mask:
        
        Return:
            recontruct loss and kl loss
        """
        bs,_,jn,_ = future_pose.shape
        B, S, N, _ = pred_pose.shape
        jidx = np.setdiff1d(np.arange(jn),[rootidx])

        #[bs, nk, fn, 21, 3]
        # print(jidx.shape, future_pose.shape, pred_pose.shape)
        rec_body_pose_loss = torch.mean((future_pose[:,:,jidx] - pred_pose[:,:,jidx]).pow(2).sum(dim=-1))
        rec_trans_loss = (future_pose[:,:,rootidx]-pred_pose[:,:,rootidx]).pow(2).sum(dim=-1).mean()
        # print(pred_pose[:,:,jidx].shape, pred_pose[:,:,rootidx].shape)
        pred_pose_global = pred_pose.clone()
        pred_pose_global[:,:,jidx] = pred_pose_global[:,:,jidx] + pred_pose[:,:,rootidx].reshape(B,S,1,3)

        rec_hsdf = self._get_hsdf(pred_pose_global,scene_points)
        rec_sdf = self.get_sdf_grid_batch(pred_pose_global,scene_sdf, scene_index)

        rec_hsdf_loss = F.l1_loss(rec_hsdf, future_hsdf)
        rec_sdf_loss = F.l1_loss(rec_sdf, future_msdf)
        rec_sdf_err = torch.mean(torch.abs(rec_sdf - future_msdf), dim = (0,2))
        rec_hsdf_err = torch.mean(torch.abs(rec_hsdf - future_hsdf), dim = (0,2))
        
        return  rec_trans_loss, rec_body_pose_loss, rec_sdf_loss, rec_hsdf_loss, rec_sdf_err, rec_hsdf_err



    def _rotate_crop_sdf(self,location, scene_sdf, index1,index2,R, sdf_len=100):

        batch_size = location.shape[0]
        location = location.reshape(batch_size,3)


        # Define the volume's parameters
        resolution = 0.05
        length = 5.0
        half_length = length / 2
        num_points = int(length / resolution)

        # Generate a range of offsets with the given resolution
        offsets = torch.linspace(-half_length, half_length-resolution, num_points)

        # Create a 3D meshgrid for the x, y, z coordinates
        X, Y, Z = torch.meshgrid(offsets, offsets, offsets, indexing='ij')

        # Adjust by the center point values
        X = X + location[:, 0, None, None, None]
        Y = Y + location[:, 1, None, None, None]
        Z = Z + location[:, 2, None, None, None]

        # Stack the coordinates to get the [batch_size, 100, 100, 100, 3] shape
        volume = torch.stack((X, Y, Z), axis=-1).cuda().float()
        location = location.cuda()
        index = torch.zeros(batch_size,6).cuda()
        index[:,:3] = volume[:,0,0,0,:]
        index[:,3:] = volume[:,99,99,99,:]
        index[:,:3] -= location
        index[:,3:] -= location
        # Reshape the volume for transformation
        volume = volume.view(batch_size, -1, 3).transpose(1, 2)
        R_inverse = torch.linalg.inv(R)
        volume = torch.matmul(R_inverse, volume)
        volume = volume.transpose(1, 2).view(batch_size, num_points**3, 3)
        volume = 2*((volume - index1[:,None,:])/(index2[:,None,:] - index1[:,None,:]) - 0.5)
        # print(volume)

            # Mask out-of-range coordinates
        out_of_range_mask = (volume > 1) | (volume < -1)
        out_of_range_rows = torch.any(out_of_range_mask, dim=2)
        out_of_range_indices = torch.where(out_of_range_rows)

        # Set the values of the out-of-range points to 0
        volume[out_of_range_indices] = 0
        scene_sdf = scene_sdf.float()
        volume = volume.float()
        sdf_volume = F.grid_sample(scene_sdf[:,None],volume[:,None,None,:,[2,1,0]], align_corners=True)

        sdf_volume = sdf_volume.view(batch_size, num_points**3)
        # sdf_volume[out_of_range_indices[0], out_of_range_indices[1]] = 0
        sdf_volume[out_of_range_indices] = 0
        sdf_volume = sdf_volume.view(batch_size, num_points, num_points, num_points).float()
 

        return sdf_volume, index



    def _train(self, train_dataloader: DataLoader, epoch_id: int):
        phase = 'train'
        self.log[phase][epoch_id] = defaultdict(list)
        rec_sdf_err = torch.zeros(60).cuda()
        rec_hsdf_err = torch.zeros(60).cuda()
        cont = 0
        dct_n = 90
        dct_m,_ = get_dct_matrix(90)
        pad_idx = np.repeat([30-1],60)
        i_idx = np.append(np.arange(0,30),pad_idx)
        for data in tqdm(train_dataloader):

            ## unpack data
            # [pose, scene_sdf, scene_origin, item_key, index, msdf_input, hsdf_input, msdf_in, hsdf_in] = data
            [pose_rotate, scene_origin_rotate, item_key, scene_sdf_origin, index1, index2, rotation_matrix] = data
            B, S, N, _ = pose_rotate.shape
            scene_sdf_origin = scene_sdf_origin.cuda().float()
            
            
            index1 = index1.cuda()
            index2 = index2.cuda()
            rotation_matrix = rotation_matrix.cuda()

            scene_sdf, index = self._rotate_crop_sdf(scene_origin_rotate,scene_sdf_origin,index1,index2,rotation_matrix,sdf_len=100)
            pose = pose_rotate.cuda().float()



            dct_m_in = torch.tensor(dct_m).cuda().unsqueeze(0).repeat(B,1,1).float()
            msdf_input = self.get_sdf_grid_batch(pose,scene_sdf,index)
            hsdf_input = self._get_hsdf(pose,self.scene_points1)
            hsdf_in = torch.bmm(dct_m_in[:,:dct_n,:],hsdf_input[:,i_idx,:])
            msdf_in = torch.bmm(dct_m_in[:,:dct_n,:], msdf_input[:,i_idx,:])
            hsdf_in = hsdf_in.transpose(1,2)
            msdf_in = msdf_in.transpose(1,2)
            

            with torch.no_grad():
                bs = pose.shape[0]
                nj = pose.shape[2]
                joints_orig = pose[:, :, 14:15].clone()
                pose = pose - joints_orig
                pose[:, :, 14:15] = joints_orig
            
            hlen = self.config.history_len
            future_motion_sdf = msdf_input[:,hlen:,:]

            history_pose = pose[:,:hlen,:,:]
            future_pose = pose[:,hlen:,:,:]

            

        
            future_hsdf = hsdf_input[:,hlen:,:]
            pred_body, pred_msdf, pred_hsdf = self._forward(
                history_pose,scene_sdf,msdf_in,hsdf_in
            )

            pred_msdf_loss = F.l1_loss(pred_msdf, msdf_input)
            pred_hsdf_loss = F.l1_loss(pred_hsdf, hsdf_input)
            forward_time = time.time()
            [ rec_trans_loss, rec_body_pose_loss, rec_sdf_loss, rec_hsdf_loss, rec_sdf_err, rec_hsdf_err] = self._cal_loss(
                future_pose, pred_body, future_hsdf, future_motion_sdf, self.rootidx, scene_sdf, index, self.scene_points
            )
         
            rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
                        self.config.weight_loss_rec_body_pose * rec_body_pose_loss
            loss = self.config.weight_loss_sdf * rec_sdf_loss + \
            rec_loss + rec_hsdf_loss + pred_msdf_loss + pred_hsdf_loss

            ## backward
            self.optimizer_h.zero_grad()
            loss.backward()
            self.optimizer_h.step()
            self.log[phase][epoch_id]['loss'].append(loss.item())
            self.log[phase][epoch_id]['rec_loss'].append(rec_loss.item())
            self.log[phase][epoch_id]['rec_trans_loss'].append(rec_trans_loss.item())
            self.log[phase][epoch_id]['rec_orient_loss'].append(0)
            self.log[phase][epoch_id]['rec_body_pose_loss'].append(rec_body_pose_loss.item())
            self.log[phase][epoch_id]['rec_hand_pose_loss'].append(0)
            self.log[phase][epoch_id]['rec_sdf_loss'].append(rec_sdf_loss.item())
            self.log[phase][epoch_id]['rec_hsdf_loss'].append(rec_hsdf_loss.item())
            # self.log[phase][epoch_id]['smooth_loss1'].append(smooth_loss1.item())
        self.scheduler.step()
        my_lr = self.scheduler.optimizer.param_groups[0]['lr']
        print('lr',my_lr)
    
    def _val(self, val_dataloader: DataLoader, epoch_id: int):
        phase = 'val'
        self.log[phase][epoch_id] = defaultdict(list)

        sdf_err = torch.zeros(90).cuda()
        hsdf_err = torch.zeros(90).cuda()
        rec_sdf_err = torch.zeros(60).cuda()
        rec_hsdf_err = torch.zeros(60).cuda()
        cont = 0
        dct_n = 90
        dct_m,_ = get_dct_matrix(90)
        pad_idx = np.repeat([30-1],60)
        i_idx = np.append(np.arange(0,30),pad_idx)
        for data in tqdm(val_dataloader):

            ## unpack data
            # [pose, scene_sdf, scene_origin, item_key, index, msdf_input, hsdf_input, msdf_in, hsdf_in] = data
            [pose_rotate, scene_origin_rotate, item_key, scene_sdf_origin, index1, index2, rotation_matrix] = data
            B, S, N, _ = pose_rotate.shape
            scene_sdf_origin = scene_sdf_origin.cuda().float()
            
            
            index1 = index1.cuda()
            index2 = index2.cuda()
            rotation_matrix = rotation_matrix.cuda()

            scene_sdf, index = self._rotate_crop_sdf(scene_origin_rotate,scene_sdf_origin,index1,index2,rotation_matrix,sdf_len=100)
            pose = pose_rotate.cuda().float()

            dct_m_in = torch.tensor(dct_m).cuda().unsqueeze(0).repeat(B,1,1).float()
            msdf_input = self.get_sdf_grid_batch(pose,scene_sdf,index)
            hsdf_input = self._get_hsdf(pose,self.scene_points1)
            hsdf_in = torch.bmm(dct_m_in[:,:dct_n,:],hsdf_input[:,i_idx,:])
            msdf_in = torch.bmm(dct_m_in[:,:dct_n,:], msdf_input[:,i_idx,:])
            hsdf_in = hsdf_in.transpose(1,2)
            msdf_in = msdf_in.transpose(1,2)
            

            scene_sdf = scene_sdf.cuda()
            pose = pose.cuda()
            index = index.cuda()


            with torch.no_grad():
                # msdf_input = self.get_sdf_grid_batch(pose,scene_sdf, index)
                # hsdf_input = self._get_hsdf(pose, self.scene_points1)
                bs = pose.shape[0]
                nj = pose.shape[2]
                joints_orig = pose[:, :, 14:15].clone()
                pose = pose - joints_orig
                pose[:, :, 14:15] = joints_orig
            
            # print('1', pose.shape)
            # print(scene_trans.shape,motion_transformation.shape,scene_sdf.shape,trans.shape,orient.shape,pose_body.shape,pose_hand.shape)
            hlen = self.config.history_len
            # print("future orient", future_orient.shape) 64 25 6
            ## forward
            future_motion_sdf = msdf_input[:,hlen:,:]

            history_pose = pose[:,:hlen,:,:]
            future_pose = pose[:,hlen:,:,:]

            

        
            future_hsdf = hsdf_input[:,hlen:,:]
            pred_body, pred_msdf, pred_hsdf = self._forward(
                history_pose,scene_sdf,msdf_in,hsdf_in
            )

            pred_msdf_loss = F.l1_loss(pred_msdf, msdf_input)
            pred_hsdf_loss = F.l1_loss(pred_hsdf, hsdf_input)
            sdf_err = sdf_err + torch.mean(torch.abs(pred_msdf - msdf_input),dim=(0,2))
            hsdf_err = hsdf_err + torch.mean(torch.abs(pred_hsdf - hsdf_input),dim=(0,2))
            cont = cont + 1

            forward_time = time.time()
            [ rec_trans_loss, rec_body_pose_loss, rec_sdf_loss, rec_hsdf_loss, rec_sdf_e, rec_hsdf_e] = self._cal_loss(
                future_pose, pred_body, future_hsdf, future_motion_sdf, self.rootidx, scene_sdf, index, self.scene_points
            )
            rec_sdf_err = rec_sdf_err + rec_sdf_e
            rec_hsdf_err = rec_hsdf_err + rec_hsdf_e
            rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
                        self.config.weight_loss_rec_body_pose * rec_body_pose_loss
            loss = self.config.weight_loss_sdf * rec_sdf_loss + \
            rec_loss + rec_hsdf_loss + 0.5*pred_msdf_loss + 0.5*pred_hsdf_loss

            path_error = torch.mean((pred_body[:,:,self.rootidx,:] - future_pose[:,:,self.rootidx,:]).norm(dim=-1),dim=0)
            bs,_,jn,_ = future_pose.shape
            jidx = np.setdiff1d(np.arange(jn),[self.rootidx])
            pose_error = torch.mean((pred_body[:,:,jidx,:] - future_pose[:,:,jidx,:]).norm(dim=-1),dim=(0,2))

            self.log[phase][epoch_id]['loss'].append(loss.item())
            self.log[phase][epoch_id]['rec_loss'].append(rec_loss.item())
            self.log[phase][epoch_id]['rec_trans_loss'].append(rec_trans_loss.item())
            self.log[phase][epoch_id]['rec_orient_loss'].append(0)
            self.log[phase][epoch_id]['rec_body_pose_loss'].append(rec_body_pose_loss.item())
            self.log[phase][epoch_id]['rec_hand_pose_loss'].append(0)
            self.log[phase][epoch_id]['rec_sdf_loss'].append(rec_sdf_loss.item())
            self.log[phase][epoch_id]['rec_hsdf_loss'].append(rec_hsdf_loss.item())
            self.log[phase][epoch_id]['path_error'].append(path_error.detach().cpu().numpy())
            self.log[phase][epoch_id]['pose_error'].append(pose_error.detach().cpu().numpy())
            # self.log[phase][epoch_id]['smooth_loss1'].append(smooth_loss1.item())
    
        # sdf_err = sdf_err / cont
        # hsdf_err = hsdf_err / cont
        # rec_sdf_err = rec_sdf_err / cont
        # rec_hsdf_err = rec_hsdf_err / cont
        # sdf_err = sdf_err.detach().cpu().numpy()
        # hsdf_err = hsdf_err.detach().cpu().numpy()
        # rec_sdf_err = rec_sdf_err.detach().cpu().numpy()
        # rec_hsdf_err = rec_hsdf_err.detach().cpu().numpy()
        # sdf_err = sdf_err[30:]
        # hsdf_err = hsdf_err[30:]
        # # Convert arrays to DataFrame
        # df = pd.DataFrame({'sdf': sdf_err, 'dis': hsdf_err,'rec_sdf':rec_sdf_err,'rec_dis':rec_hsdf_err})

        # # Save DataFrame to CSV file
        # df.to_csv('pred_muld.csv', index=False)
        ## ckeck best
        cur_criterion = 'rec_loss'
        cur_best = np.mean(self.log[phase][epoch_id][cur_criterion])
        if cur_best < self.best[cur_criterion]:
            for key in self.best:
                if key != 'epoch':
                    self.best[key] = np.mean(self.log[phase][epoch_id][key])
            self.best['epoch'] = epoch_id
            
            ## save best
            self._save_state_dict(epoch_id, 'model_best')
        
        # save every 5 epoch
        if epoch_id % 5 == 0:
            self._save_state_dict(epoch_id, 'epoch{}'.format(epoch_id))
    







    
    def _epoch_report_val(self, epoch_id: int):
        Console.log("epoch [{}/{}] done...".format(epoch_id+1, self.config.num_epoch))
        path_err = self.log['val'][epoch_id]['path_error']
        # print('patherr', path_err)
        path_err = np.stack(path_err, axis = 0)
        pose_err = self.log['val'][epoch_id]['pose_error']
        pose_err = np.stack(pose_err, axis = 0)
        
        path_err = np.mean(path_err, axis = 0).reshape(60)
        pose_err = np.mean(pose_err, axis = 0).reshape(60)
        Console.log("0.5s path:{:.5f}, 1.0s path:{:.5f}, 1.5s path:{:.5f}, 2.0s path:{:.5f}".format(path_err[14], path_err[29], path_err[44], path_err[59]))
        Console.log("0.5s pose:{:.5f}, 1.0s pose:{:.5f}, 1.5s pose:{:.5f}, 2.0s pose:{:.5f}".format(pose_err[14], pose_err[29], pose_err[44], pose_err[59]))
        
        epoch_report_str = EPOCH_VAL_REPORT_TEMPLATE.format(
            val_total_loss=round(np.mean(self.log['val'][epoch_id]['loss']), 5),
            val_rec_loss=round(np.mean(self.log['val'][epoch_id]['rec_loss']), 5),
            val_rec_trans_loss=round(np.mean(self.log['val'][epoch_id]['rec_trans_loss']), 5),
            val_rec_orient_loss=round(np.mean(self.log['val'][epoch_id]['rec_orient_loss']), 5),
            val_rec_body_pose_loss=round(np.mean(self.log['val'][epoch_id]['rec_body_pose_loss']), 5),
            val_rec_hand_pose_loss=round(np.mean(self.log['val'][epoch_id]['rec_hand_pose_loss']), 5),
            val_rec_sdf_loss=round(np.mean(self.log['val'][epoch_id]['rec_sdf_loss']), 5),
            val_rec_hsdf_loss=round(np.mean(self.log['val'][epoch_id]['rec_hsdf_loss']), 5),
            val_mpjpe=round(np.mean(self.log['val'][epoch_id]['mpjpe']), 5),
            val_mpvpe=round(np.mean(self.log['val'][epoch_id]['mpvpe']), 5),
            val_pose_error=round(np.mean(self.log['val'][epoch_id]['pose_error']), 5),
            val_path_error=round(np.mean(self.log['val'][epoch_id]['path_error']), 5),
            # val_rec_sdf_loss=round(np.mean(self.log['val'][epoch_id]['smooth_loss1']), 5),
        )
        Console.log(epoch_report_str)




    
    def _epoch_report(self, epoch_id: int):
        Console.log("epoch [{}/{}] done...".format(epoch_id+1, self.config.num_epoch))

        path_err = self.log['val'][epoch_id]['path_error']
        path_err = np.stack(path_err, axis = 0)
        pose_err = self.log['val'][epoch_id]['pose_error']
        pose_err = np.stack(pose_err, axis = 0)
        
        path_err = np.mean(path_err, axis = 0).reshape(60)
        pose_err = np.mean(pose_err, axis = 0).reshape(60)
        Console.log("0.5s path:{:.5f}, 1.0s path:{:.5f}, 1.5s path:{:.5f}, 2.0s path:{:.5f}".format(path_err[14], path_err[29], path_err[44], path_err[59]))
        Console.log("0.5s pose:{:.5f}, 1.0s pose:{:.5f}, 1.5s pose:{:.5f}, 2.0s pose:{:.5f}".format(pose_err[14], pose_err[29], pose_err[44], pose_err[59]))

        
        epoch_report_str = EPOCH_REPORT_TEMPLATE.format(
            train_total_loss=round(np.mean(self.log['train'][epoch_id]['loss']), 5),
            train_rec_loss=round(np.mean(self.log['train'][epoch_id]['rec_loss']), 5),
            train_rec_trans_loss=round(np.mean(self.log['train'][epoch_id]['rec_trans_loss']), 5),
            train_rec_orient_loss=round(np.mean(self.log['train'][epoch_id]['rec_orient_loss']), 5),
            train_rec_body_pose_loss=round(np.mean(self.log['train'][epoch_id]['rec_body_pose_loss']), 5),
            train_rec_hand_pose_loss=round(np.mean(self.log['train'][epoch_id]['rec_hand_pose_loss']), 5),
            train_rec_sdf_loss=round(np.mean(self.log['train'][epoch_id]['rec_sdf_loss']), 5),
            val_total_loss=round(np.mean(self.log['val'][epoch_id]['loss']), 5),
            val_rec_loss=round(np.mean(self.log['val'][epoch_id]['rec_loss']), 5),
            val_rec_trans_loss=round(np.mean(self.log['val'][epoch_id]['rec_trans_loss']), 5),
            val_rec_orient_loss=round(np.mean(self.log['val'][epoch_id]['rec_orient_loss']), 5),
            val_rec_body_pose_loss=round(np.mean(self.log['val'][epoch_id]['rec_body_pose_loss']), 5),
            val_rec_hand_pose_loss=round(np.mean(self.log['val'][epoch_id]['rec_hand_pose_loss']), 5),
            val_rec_sdf_loss=round(np.mean(self.log['val'][epoch_id]['rec_sdf_loss']), 5),
            val_rec_hsdf_loss=round(np.mean(self.log['val'][epoch_id]['rec_hsdf_loss']), 5),
            val_mpjpe=round(np.mean(self.log['val'][epoch_id]['mpjpe']), 5),
            val_mpvpe=round(np.mean(self.log['val'][epoch_id]['mpvpe']), 5),
            val_pose_error=round(np.mean(self.log['val'][epoch_id]['pose_error']), 5),
            val_path_error=round(np.mean(self.log['val'][epoch_id]['path_error']), 5),
           
        )
        Console.log(epoch_report_str)
    
    def _best_report(self):
        Console.log("training completed...")

        best_report_str = BEST_REPORT_TEMPLATE.format(
            best_epoch=self.best['epoch'],
            best_total_loss=self.best['loss'],
            best_rec_loss=self.best['rec_loss'],
            best_rec_trans_loss=self.best['rec_trans_loss'],
            best_rec_orient_loss=self.best['rec_orient_loss'],
            best_rec_body_pose_loss=self.best['rec_body_pose_loss'],
            best_rec_hand_pose_loss=self.best['rec_hand_pose_loss'],
            best_rec_sdf_loss=self.best['rec_sdf_loss']
            # best_rec_sdf_loss=self.best['smooth_loss1']
        )
        Console.log(best_report_str)

    def _dump_log(self, epoch_id: int):
        dump_logs = {}

        for key in self.dump_keys:
            k = 'train/' + key
            dump_logs[k] = {
                'plot': True,
                'step': epoch_id,
                'value': np.mean(self.log['train'][epoch_id][key])
            }
        for key in self.dump_keys:
            k = 'val/' + key
            dump_logs[k] = {
                'plot': True,
                'step': epoch_id,
                'value': np.mean(self.log['val'][epoch_id][key])
            }

        Ploter.write(dump_logs)
    
    