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
from model.basemodel import MotionGRU, SceneNet, SDFNet
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

        # self.cond_net = CondNet(self.config).to(self.config.device)
        # self.base_model = MotionModel(self.config).to(self.config.device)

        # # to be done
        self.scene_net = SceneNet(self.config).to(self.config.device)
        # self.motion_net = MotionNet(self.config).to(self.config.device)
        self.sdf_net = SDFNet(self.config).to(self.config.device)
        self.motion_decoder = MotionGRU(self.config).to(self.config.device)
        self.dataloader = dataloader
        scene_point_path = config.dataset_scene_points
        with open(scene_point_path,'rb') as f:
            data = pickle.load(f)
        data = torch.FloatTensor(data)
        # self.scene_points = torch.FloatTensor(data)

        
        with open(config.dataset_scene_points, 'rb') as f:
            data = pickle.load(f)

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
                {'params': list(self.scene_net.parameters())},
                {'params': list(self.motion_decoder.parameters())},
                {'params': list(self.sdf_net.parameters())}
                
            ],
            lr = self.config.lr
        )


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
    
    def _save_state_dict(self, epoch: int, name: str):
        # saved_cond_net_state_dict = {k: v for k, v in self.cond_net.state_dict().items() if 'bert' not in k} # don't save bert weights
        torch.save({
            'epoch': epoch + 1,
            'scene_net_state_dict': self.scene_net.state_dict(),
            'sdf_net_state_dict': self.sdf_net.state_dict(),
            'motion_decoder_state_dict': self.motion_decoder.state_dict(),
            'optimizer_h_state_dict': self.optimizer_h.state_dict()
        }, os.path.join(self.config.log_dir, '{}.pth'.format(name)))

    def _load_state_dict(self):
        if os.path.isdir(self.config.resume_model):
            ckp_file = os.path.join(self.config.resume_model, 'epoch40.pth')
        elif os.path.isfile(self.config.resume_model):
            ckp_file = self.config.resume_model
        else:
            return 0
     
        print("the ckp file is ", ckp_file)
        state_dict = torch.load(ckp_file)
        # ## load cond net weight
        # cond_net_dict = self.cond_net.state_dict()
        # cond_net_dict.update(state_dict['cond_net_state_dict'])
        # self.cond_net.load_state_dict(cond_net_dict)
        # ## load cvae model weight
        # self.base_model.load_state_dict(state_dict['base_model_state_dict'])

        self.motion_decoder.load_state_dict(state_dict['motion_decoder_state_dict'])
        self.sdf_net.load_state_dict(state_dict['sdf_net_state_dict'])
        self.scene_net.load_state_dict(state_dict['scene_net_state_dict'])
        Console.log('Load checkpoint: {}'.format(ckp_file))
        return state_dict['epoch']
    
    def _set_phase(self, phase: str):
        if phase == "train":
            self.sdf_net.train()
            self.scene_net.train()
            # self.motion_net.train()
            self.motion_decoder.train()
        elif phase == "val":
            self.sdf_net.eval()
            self.scene_net.eval()
            # self.motion_net.eval()
            self.motion_decoder.eval()
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
        self, history_body, scene_sdf, motion_sdf, hsdf
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
        B, S,N, _ = history_body.shape
        # print(trans_shape,orient_shape,pose_body_shape,pose_hand_shape)
        # B S 162
        history_body = history_body.reshape(B,S,N*3)
        # print("************")
        # B S 162
        # print(history_motion[:,-1,:3])
        history_motion = history_body.permute(1,0,2)

        
        # print(history_motion[-1,:,:3])
        # print("----------------")
        # print("input shape", history_motion.shape) 25 64 162
        scene_sdf = scene_sdf.unsqueeze(1)
        fs = self.scene_net(scene_sdf)
        fsdf = self.sdf_net(motion_sdf)
        hsdf = hsdf.permute(1,0,2)
        fsdf = fsdf.permute(1,0,2)
        output = self.motion_decoder(history_motion, fs, fsdf, hsdf)
        output = output.permute(1,0,2)
        output = output.reshape(B,60,N,3)
        # print("output",output.shape)
        pred_body = output
        return pred_body

    

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

        
        return  rec_trans_loss, rec_body_pose_loss, rec_sdf_loss, rec_hsdf_loss




    def _train(self, train_dataloader: DataLoader, epoch_id: int):
        phase = 'train'
        self.log[phase][epoch_id] = defaultdict(list)

        for data in tqdm(train_dataloader):

            ## unpack data
            # [pose, scene_sdf, scene_origin, item_key, index, msdf_input, hsdf_input, msdf_in, hsdf_in] = data
            [pose, scene_sdf, scene_origin, item_key, index] = data
            B, S, N, _ = pose.shape




            scene_sdf = scene_sdf.cuda()
            pose = pose.cuda()
            index = index.cuda()
            
            # msdf_input = msdf_input.cuda()
            # hsdf_input = hsdf_input.cuda()
            with torch.no_grad():
                msdf_input = self.get_sdf_grid_batch(pose,scene_sdf, index)
                hsdf_input = self._get_hsdf(pose, self.scene_points1)
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
            # print("-----------------")
            # future_motion_sdf = self._get_motion_sdf(future_trans,future_orient,future_pose_body,future_pose_hand,betas,scene_sdf,scene_radius, True)
            # print(future_motion_sdf.shape)
           
            pred_body = self._forward(
                history_pose,scene_sdf,msdf_input,hsdf_input
            )
            # print('2', pose.shape)
            # print(pred_trans.shape)
            forward_time = time.time()
            [ rec_trans_loss, rec_body_pose_loss, rec_sdf_loss, rec_hsdf_loss] = self._cal_loss(
                future_pose, pred_body, future_hsdf, future_motion_sdf, self.rootidx, scene_sdf, index, self.scene_points
            )
         
            # rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
            #             self.config.weight_loss_rec * rec_orient_loss + \
            #             self.config.weight_loss_rec_body_pose * rec_body_pose_loss + \
            #             self.config.weight_loss_rec_hand_pose * rec_hand_pose_loss
            # loss = self.config.weight_loss_sdf * rec_sdf_loss + \
            #         rec_loss
            rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
                        self.config.weight_loss_rec_body_pose * rec_body_pose_loss
            loss = self.config.weight_loss_sdf * rec_sdf_loss + \
            rec_loss + rec_hsdf_loss

            ## backward
            self.optimizer_h.zero_grad()
            loss.backward()
            self.optimizer_h.step()
            # back_time =time.time()

            # print("motion sdf", motion_sdf_time- start, "forward", forward_time-motion_sdf_time,"loss",cal_loss_time-forward_time,"back",back_time-cal_loss_time)
            ## record log
            # iter_time = time.time() - start
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
        
        for data in tqdm(val_dataloader):
            start = time.time()
            ## unpack data
            # [pose, scene_sdf, scene_origin, item_key, index, msdf_input, hsdf_input, msdf_in, hsdf_in] = data
            [pose, scene_sdf, scene_origin, item_key, index] = data
            B, S, N, _ = pose.shape




            scene_sdf = scene_sdf.cuda()
            pose = pose.cuda()
            index = index.cuda()
            
            # msdf_input = msdf_input.cuda()
            # hsdf_input = hsdf_input.cuda()
            with torch.no_grad():
                msdf_input = self.get_sdf_grid_batch(pose,scene_sdf, index)
                hsdf_input = self._get_hsdf(pose, self.scene_points1)
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
            # print("-----------------")
            # future_motion_sdf = self._get_motion_sdf(future_trans,future_orient,future_pose_body,future_pose_hand,betas,scene_sdf,scene_radius, True)
            # print(future_motion_sdf.shape)
           
            pred_body = self._forward(
                history_pose,scene_sdf,msdf_input,hsdf_input
            )
            # print('2', pose.shape)
            # print(pred_trans.shape)
            forward_time = time.time()
            [ rec_trans_loss, rec_body_pose_loss, rec_sdf_loss, rec_hsdf_loss] = self._cal_loss(
                future_pose, pred_body, future_hsdf, future_motion_sdf, self.rootidx, scene_sdf, index, self.scene_points
            )
         
            # rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
            #             self.config.weight_loss_rec * rec_orient_loss + \
            #             self.config.weight_loss_rec_body_pose * rec_body_pose_loss + \
            #             self.config.weight_loss_rec_hand_pose * rec_hand_pose_loss
            # loss = self.config.weight_loss_sdf * rec_sdf_loss + \
            #         rec_loss
            rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
                        self.config.weight_loss_rec_body_pose * rec_body_pose_loss
            loss = self.config.weight_loss_sdf * rec_sdf_loss + \
            rec_loss + rec_hsdf_loss


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
        # print(self.log['val'][epoch_id]['path_error'])
        # print(self.log['val'][epoch_id]['mpjpe'])
        # print(self.log['val'][epoch_id]['pose_error'])
        Console.log("epoch [{}/{}] done...".format(epoch_id+1, self.config.num_epoch))
        path_err = self.log['val'][epoch_id]['path_error']
        # print('patherr', path_err)
        path_err = np.stack(path_err, axis = 0)
        print(path_err.shape)
        pose_err = self.log['val'][epoch_id]['pose_error']
        pose_err = np.stack(pose_err, axis = 0)
        
        print(pose_err.shape)
        path_err = np.mean(path_err, axis = 0).reshape(60)
        pose_err = np.mean(pose_err, axis = 0).reshape(60)
        # print("0.5s, path error", path_err[14])
        # print("1s, path_error", path_err[29])
        # print("0.5s, pose error", pose_err[14])
        # print("1s, pose_error", pose_err[29])
        # print("mean, path_error", path_err.mean())
        # print("mean, pose_error", pose_err.mean())
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

        
        epoch_report_str = EPOCH_REPORT_TEMPLATE.format(
            train_total_loss=round(np.mean(self.log['train'][epoch_id]['loss']), 5),
            train_rec_loss=round(np.mean(self.log['train'][epoch_id]['rec_loss']), 5),
            train_rec_trans_loss=round(np.mean(self.log['train'][epoch_id]['rec_trans_loss']), 5),
            train_rec_orient_loss=round(np.mean(self.log['train'][epoch_id]['rec_orient_loss']), 5),
            train_rec_body_pose_loss=round(np.mean(self.log['train'][epoch_id]['rec_body_pose_loss']), 5),
            train_rec_hand_pose_loss=round(np.mean(self.log['train'][epoch_id]['rec_hand_pose_loss']), 5),
            train_rec_sdf_loss=round(np.mean(self.log['train'][epoch_id]['rec_sdf_loss']), 5),
            # train_rec_sdf_loss=round(np.mean(self.log['train'][epoch_id]['smooth_loss1']), 5),
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
    
    