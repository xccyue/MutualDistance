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
from model.basemodel import MotionGRU_S1, SceneNet, GCN_H, GCN_S
from utils.utilities import Console, Ploter
from utils.visualization import render_attention, frame2video, render_reconstructed_motion_in_scene, render_sample_k_motion_in_scene,render_motion_in_scene, render_motion_point_in_scene
import utils.configuration as config
import trimesh
from utils.model_utils import GeometryTransformer
# from human_body_prior.tools.model_loader import load_model
# from human_body_prior.models.vposer_model import VPoser
from utils.smplx_util import SMPLX_Util, marker_indic_62
import smplx
from utils.geo_utils import smplx_signed_distance
import pickle
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from utils.data_utils import get_dct_matrix


EPOCH_REPORT_TEMPLATE = """
----------------------summary----------------------
[train] train_sdf_loss: {train_sdf_loss}
[train] train_hsdf_loss: {train_hsdf_loss}
[val] val_sdf_loss: {val_sdf_loss}
[val] val_hsdf_loss: {val_hsdf_loss}
"""
EPOCH_VAL_REPORT_TEMPLATE = """
----------------------summary----------------------
[val] val_sdf_loss: {val_sdf_loss}
[val] val_hsdf_loss: {val_hsdf_loss}
"""

BEST_REPORT_TEMPLATE = """
----------------------best----------------------
[best] best epoch: {best_epoch}
[best] best_total_loss: {best_sdf_loss}
"""

def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


# mark = config.marker_file
# print(mark)
# marker_indic = []
# with open (mark,'rb') as f:
#     for line in f.readlines():
#         marker_indic.append(int(line))
# print(len(marker_indic))

# overlap_maker_indic = []

# for inc in marker_indic_62:
#     for id, inc1 in enumerate(marker_indic):
#         if(inc == inc1):
#             overlap_maker_indic.append(id)

# print("There are ", len(overlap_maker_indic), "in overlap set")


class MotionSolver():
    def __init__(self, conf: Any, dataloader: dict):
        self.config = conf


        self.scene_net = SceneNet(self.config).to(self.config.device)
        self.sdf_gcn = GCN_S(self.config).to(self.config.device)
        self.hsdf_gcn = GCN_H(self.config).to(self.config.device)
        self.motion_encoder = MotionGRU_S1(self.config).to(self.config.device)
        self.dataloader = dataloader
        self.dct_n = config.dct_n


        dct_n = 90
        dct_m_in, _ = get_dct_matrix(30 + 60)
        dct_m_out, _ = get_dct_matrix(30 + 60)
        pad_idx = np.repeat([30 - 1], 60)
        i_idx = np.append(np.arange(0,30),pad_idx)
        self.batch = config.batch_size
        self.dct_m_in = dct_m_in
        self.i_idx = i_idx

        with open(config.dataset_scene_points, 'rb') as f:
            data = pickle.load(f)

        scene_points = torch.tensor(data).cuda().float()
        self.scene_points = scene_points.unsqueeze(0).repeat(config.batch_size*60,1,1)
        self.scene_points1 = scene_points.unsqueeze(0).repeat(config.batch_size*90,1,1)
        self.rootidx = 14


        self.optimizer_h = optim.Adam(
            [
                {'params': list(self.scene_net.parameters())},
                {'params': list(self.sdf_gcn.parameters())},
                {'params': list(self.hsdf_gcn.parameters())},
                {'params': list(self.motion_encoder.parameters())}
                
            ],
            lr = self.config.lr
        )

        self.log = {phase: {} for phase in ["train", "val"]}
        self.dump_keys = ['sdf_loss', 'hsdf_loss', 'loss']

        self.best = {
            'loss': float("inf"),
            'epoch': 0
        }


        # self._report_model_size()

    def _save_state_dict(self, epoch: int, name: str):
        # saved_cond_net_state_dict = {k: v for k, v in self.cond_net.state_dict().items() if 'bert' not in k} # don't save bert weights
        torch.save({
            'epoch': epoch + 1,
            'scene_net_state_dict': self.scene_net.state_dict(),
            'sdf_gcn_state_dict': self.sdf_gcn.state_dict(),
            'hsdf_gcn_state_dict': self.hsdf_gcn.state_dict(),
            'motion_encoder_state_dict': self.motion_encoder.state_dict(),
            'optimizer_h_state_dict': self.optimizer_h.state_dict(),
            # 'scheduler_state_dict' : self.scheduler.state_dict()
        }, os.path.join(self.config.log_dir, '{}.pth'.format(name)))

    def _load_state_dict(self):
        if os.path.isdir(self.config.resume_model):
            ckp_file = os.path.join(self.config.resume_model, "model_best.pth")
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
        print(state_dict.keys())
        self.motion_encoder.load_state_dict(state_dict['motion_encoder_state_dict'])
        self.sdf_gcn.load_state_dict(state_dict['sdf_gcn_state_dict'])
        self.hsdf_gcn.load_state_dict(state_dict['hsdf_gcn_state_dict'])
        self.scene_net.load_state_dict(state_dict['scene_net_state_dict'])
        # self.optimizer_h.load_state_dict(state_dict=['optimizer_h_state_dict'])
        # self.scheduler.load_state_dict(state_dict=['scheduler_state_dict'])
        Console.log('Load checkpoint: {}'.format(ckp_file))
        return state_dict['epoch']
    
    def _set_phase(self, phase: str):
        if phase == "train":
            self.sdf_gcn.train()
            self.hsdf_gcn.train()
            self.scene_net.train()
            # self.motion_net.train()
            self.motion_encoder.train()
        elif phase == "val":
            self.sdf_gcn.eval()
            self.hsdf_gcn.eval()
            self.scene_net.eval()
            # self.motion_net.eval()
            self.motion_encoder.eval()
        else:
            raise Exception("Invalid phase")

    def _get_body_vertices(self, trans, orient, betas, body_pose, hand_pose):
        """ Get body vertices for regress body vertices

        Args:
            smplx paramters

        Returns:
            body vertices
        """
        torch_param = {}
        torch_param['body_pose'] = body_pose
        torch_param['betas'] = betas
        torch_param['transl'] = trans
        torch_param['global_orient'] = orient
        torch_param['left_hand_pose'] = hand_pose[:, 0:45]
        torch_param['right_hand_pose'] = hand_pose[:, 45:]

        output = self.smplx_model(return_verts=True, **torch_param)
        vertices = output.vertices
        joints = output.joints
        # faces = output.faces
        return vertices, joints

    def _cal_loss(self, pred, motion_sdf, dct_n):
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
        B, T, N = motion_sdf.shape
        _, idct_m = get_dct_matrix(T)
        idct_m = torch.FloatTensor(idct_m).cuda()

        # L B*N
        pred_t = pred.view(-1, dct_n).transpose(0,1)
        #L B*N -> T B*N -> B*N T -> B, N , T -> B T N
        pred_expmap = torch.matmul(idct_m[:, :dct_n], pred_t).transpose(0, 1).contiguous().view(-1, N,
                                                                                               T).transpose(1, 2)
        
        targ_expmap = motion_sdf.clone()

        loss = torch.mean(torch.abs(pred_expmap - targ_expmap))

        return loss
        



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
        B, S,N, _ = history_body.shape
        # print(trans_shape,orient_shape,pose_body_shape,pose_hand_shape)
        # B S 162
        history_body = history_body.reshape(B,S,N*3)
        history_motion = history_body.permute(1,0,2)
        scene_sdf = scene_sdf.unsqueeze(1)
        fs = self.scene_net(scene_sdf)
        # fs B 128
        fh = self.motion_encoder(history_motion)
        # fh B 128
        # sdf_in B N L
        # beta B 10
        # print(sdf_in.shape,fs.shape,fh.shape)
        pred_sdf = self.sdf_gcn(sdf_in,fs,fh)
        pred_hsdf = self.hsdf_gcn(hsdf_in,fs,fh)
        return pred_sdf, pred_hsdf


    
    
    


    
            
    def _compute_rec_error(self, future_trans, future_orient,future_pose_body,future_pose_hand,
                pred_trans, pred_orient, pred_pose_body, pred_pose_hand, betas,
                scene_wsdf,scene_radius, future_motion_sdf, motion_transformation):

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

        verts_gt, joints_gt = self._get_body_vertices(
            torch.zeros((B*S,3)).cuda(),
            future_orient.reshape(B * S, -1),
            betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1),
            future_pose_body.reshape(B * S, -1),
            future_pose_hand.reshape(B * S, -1)
        )
        ## 2. get rec body vertices
        # print('joints', joints_gt.shape)
        verts_rec, joints_rec = self._get_body_vertices(
            torch.zeros((B*S,3)).cuda(),
            pred_orient.reshape(B * S, -1),
            betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1),
            pred_pose_body.reshape(B * S, -1),
            pred_pose_hand.reshape(B * S, -1),
        )
        pose_error = torch.sqrt(((joints_gt - joints_rec) ** 2).sum(-1)).mean(-1)




        return rec_joints_error, rec_vertex_error, path_error, pose_error
    

    def _get_hsdf_future(self, trans,orient, betas, pose_body, pose_hand, scene_index):
        B = trans.shape[0]
        S = trans.shape[1]

        orient = GeometryTransformer.convert_to_3D_rot(orient.reshape(-1, 6)).reshape(B,S,3)
        orient = orient.reshape(B*S,orient.shape[-1])
        pose_body = pose_body.reshape(B*S,pose_body.shape[-1])
        pose_hand = pose_hand.reshape(B*S,pose_hand.shape[-1])
        betas_seq = betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1)

        trans = trans.reshape(B*S,trans.shape[-1])
        verts, _ = self._get_body_vertices(
            trans,
            orient,
            betas_seq,
            pose_body,
            pose_hand
        )

        # faces = faces.astype(int)
        # faces = torch.tensor(faces, dtype=torch.long, device='cuda')
        # face_vertices = index_vertices_by_faces(verts, faces)
        # print(self.scene_points.shape)
        # print(verts.shape)
        scene_index = scene_index.unsqueeze(1).repeat(1,S,1)
        scene_index = scene_index.reshape(B*S,-1)
        scene_index = scene_index.unsqueeze(1)
        # B*S 100 3
        # B*S 10000 3
        # distance1, index, dist_type = kaolin.metrics.trianglemesh.point_to_mesh_distance(self.scene_points+scene_index[:,:,:3], face_vertices)
        # distance1 = distance1.reshape(B,S,-1)
        # distance1 = torch.sqrt(distance1)
        distance = torch.cdist(self.scene_points1,verts)
        distance = torch.min(distance,dim=2)[0]
        distance = distance.reshape(B,S,-1)
        # print(torch.max(torch.abs(distance1 - distance)))

        return distance

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


    def _get_body_vertices_h(self, trans, orient, betas, body_pose, hand_pose):
        """ Get body vertices for regress body vertices

        Args:
            smplx paramters

        Returns:
            body vertices
        """
        torch_param = {}
        torch_param['body_pose'] = body_pose
        torch_param['betas'] = betas
        torch_param['transl'] = trans
        torch_param['global_orient'] = orient
        torch_param['left_hand_pose'] = hand_pose[:, 0:45]
        torch_param['right_hand_pose'] = hand_pose[:, 45:]

        output = self.smplx_model_h(return_verts=True, **torch_param)
        vertices = output.vertices
        joints = output.joints
    
        return vertices, joints, self.smplx_model_h.faces

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
                # msdf_input = self.get_sdf_grid_batch(pose,scene_sdf, index)
                # hsdf_input = self._get_hsdf(pose, self.scene_points1)
                bs = pose.shape[0]
                nj = pose.shape[2]
                joints_orig = pose[:, :, 14:15].clone()
                pose = pose - joints_orig
                pose[:, :, 14:15] = joints_orig
                # in_sdf = 
                # in_hsdf = 

            
        
            hlen = self.config.history_len
            future_motion_sdf = msdf_input[:,hlen:,:]

            history_pose = pose[:,:hlen,:,:]
            future_pose = pose[:,hlen:,:,:]
            future_hsdf = hsdf_input[:,hlen:,:]


            # B N L
            outputs, outputs_h = self._forward(
                history_pose, scene_sdf, msdf_in, hsdf_in
            )
            B = outputs.shape[0]
            # B N*L
            outputs = outputs.view(B,-1)
            loss_sdf = self._cal_loss(outputs, msdf_input, self.dct_n)
            B = outputs_h.shape[0]
            # B N*L
            outputs_h = outputs_h.view(B,-1)

            loss_hsdf = self._cal_loss(outputs_h, hsdf_input, self.dct_n)
            # print(pred_trans.shape)
            loss = loss_hsdf + loss_sdf
         

            ## backward
            self.optimizer_h.zero_grad()
            loss.backward()
            self.optimizer_h.step()
            ## record log
            self.log[phase][epoch_id]['sdf_loss'].append(loss_sdf.item())
            self.log[phase][epoch_id]['hsdf_loss'].append(loss_hsdf.item())
        

        # self.scheduler.step()
        lr = self.optimizer_h.param_groups[0]['lr']
        print("learning rate", lr)

    def _compute_error(self, sdf):
        # print("sdf, sdf_in", sdf.shape, sdf_in.shape)
        
        sdf_t = sdf.clone()
        for i in range(30,90):
            sdf_t[:,i,:] = sdf[:,29,:].clone()
        # for i in range(sdf.shape[0]):
        #     print("start")
        #     print(sdf_t[i,:,:2])
        #     print(sdf[i,:,:2])
        #     print("end")

        loss = torch.mean(torch.abs(sdf[:,30:,:] - sdf_t[:,30:,:]))
        return loss
        

    
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
            
            with torch.no_grad():
                bs = pose.shape[0]
                nj = pose.shape[2]
                joints_orig = pose[:, :, 14:15].clone()
                pose = pose - joints_orig
                pose[:, :, 14:15] = joints_orig

                # in_sdf = 
                # in_hsdf = 

            
        
            hlen = self.config.history_len
            future_motion_sdf = msdf_input[:,hlen:,:]

            history_pose = pose[:,:hlen,:,:]
            future_pose = pose[:,hlen:,:,:]
            future_hsdf = hsdf_input[:,hlen:,:]


            # B N L
            outputs, outputs_h = self._forward(
                history_pose, scene_sdf, msdf_in, hsdf_in
            )
            B = outputs.shape[0]
            # B N*L
            outputs = outputs.view(B,-1)
            loss_sdf = self._cal_loss(outputs, msdf_input, self.dct_n)
            B = outputs_h.shape[0]
            # B N*L
            outputs_h = outputs_h.view(B,-1)

            loss_hsdf = self._cal_loss(outputs_h, hsdf_input, self.dct_n)
            # print(pred_trans.shape)
            loss = loss_hsdf + loss_sdf
         
         

            ## record log
            self.log[phase][epoch_id]['sdf_loss'].append(loss_sdf.item())
            self.log[phase][epoch_id]['hsdf_loss'].append(loss_hsdf.item())


            loss_repeat = self._compute_error(msdf_input)


            ## record log
            self.log[phase][epoch_id]['sdf_loss'].append(loss_sdf.item())
            self.log[phase][epoch_id]['repeat_loss'].append(loss_repeat.item())
            self.log[phase][epoch_id]['hsdf_loss'].append(loss_hsdf.item())
            self.log[phase][epoch_id]['loss'].append(loss.item())

        print("repeat loss", round(np.mean(self.log['val'][epoch_id]['repeat_loss']), 5))
        # motion_sdf_diff_all = motion_sdf_diff_all / len
        # motion_sdf_diff_all = motion_sdf_diff_all.detach().cpu().numpy()
        # np.save("sdf_diff", motion_sdf_diff_all)
        ## ckeck best
        cur_criterion = 'loss'
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
    
    def _report_sdf_diff(self, train_dataloader):
        phase = 'val'

        diffs = torch.zeros(67).cuda()
        n = 0
        for data in tqdm(train_dataloader):

            ## unpack data
            [scene_id, scene_radius, scene_trans, motion_transformation, scene_sdf,  trans, orient, betas, pose_body, pose_hand, sdf_in,sdf_out,motion_sdf] = data
            # print(motion_sdf.shape)
            motion_sdf1 = motion_sdf[:,1:,:]
            motion_sdf2 = motion_sdf[:,:-1,:]
            diff = torch.mean(torch.abs(motion_sdf2 - motion_sdf1),dim = [0,1])
            # print(diff.shape)
            diffs += diff
            n += 1
        print(diffs / n)
        
        



    
    def _visualize_atten(self, save_folder, scene_id, scene_trans_mat, scene_xyz, atten_score, pred_target_object):
        scene_path = os.path.join(config.scannet_folder, '{}/{}_vh_clean_2.ply'.format(scene_id, scene_id))
        static_scene = trimesh.load(scene_path, process=False)
        static_scene.apply_transform(scene_trans_mat)

        render_attention(
            save_folder=save_folder,
            scene_mesh=static_scene,
            atten_score=atten_score,
            atten_pos=scene_xyz,
            pred_target_object=pred_target_object,
        )
    

    def _save_motion(self,val_dataloader: DataLoader, epoch_id: int, save_folder=None):
        if save_folder == None:
            save_folder = config.save_vis_folder
        phase = 'val'
        self.log[phase][epoch_id] = defaultdict(list)
        save_id = 0
        for data in tqdm(val_dataloader):

            ## unpack data
            [scene_id, scene_radius, scene_trans, motion_transformation, scene_sdf,  trans, orient, betas, pose_body, pose_hand, sdf_in,sdf_out,motion_sdf] = data
          
         
       
            # print(scene_trans.shape,motion_transformation.shape,scene_sdf.shape,trans.shape,orient.shape,pose_body.shape,pose_hand.shape)
            hlen = self.config.history_len
            # print("hlen", hlen) 25
            history_trans = trans[:,:hlen,:]
            history_orient = orient[:,:hlen,:]
            history_pose_body = pose_body[:,:hlen,:]
            history_pose_hand = pose_hand[:,:hlen,:]

            future_motion_sdf = motion_sdf[:,hlen:,:]

            # B N L
            outputs = self._forward(
                history_trans, history_orient, history_pose_body, history_pose_hand,betas, scene_sdf, sdf_in
            )
            B = outputs.shape[0]
            outputs = outputs.view(B,-1)
            B, T, N = motion_sdf.shape
            _, idct_m = get_dct_matrix(T)
            idct_m = torch.FloatTensor(idct_m).cuda()

            # L B*N
            pred_t = outputs.view(-1, self.dct_n).transpose(0,1)
            #L B*N -> T B*N -> B*N T -> B, N , T -> B T N
            pred_expmap = torch.matmul(idct_m[:, :self.dct_n], pred_t).transpose(0, 1).contiguous().view(-1, N,
                                                                                                T).transpose(1, 2)
            
            targ_expmap = motion_sdf.clone()


            ## record log
            for i in range(B):
                sid = scene_id[i]

                scene_path =  "/hdd/Pointnet2.ScanNet/possion_reconstruction/scene/{}_00/{}_00_vh_clean_2.ply".format(sid,sid)
                static_scene = trimesh.load(scene_path, process=False)
                t = trans[i,:,:].cpu().detach().numpy()
                o = orient[i,:,:].cpu().detach().numpy()
                pb = pose_body[i,:,:].cpu().detach().numpy()
                pd = pose_hand[i,:,:].cpu().detach().numpy()
                o = GeometryTransformer.convert_to_3D_rot(orient[i,:,:].detach()).cpu().numpy()
                b = betas[i].detach().cpu().numpy()
             
                mt = motion_transformation[i,:].cpu().detach().numpy()
                pkl = (t+mt,o,b,pb,pd)
                name = str(save_id)
                save_id += 1
                saveid_path = os.path.join(save_folder,str(save_id))
                if not os.path.exists(saveid_path):
                    os.mkdir(saveid_path)
                render_motion_in_scene(
                    smplx_folder= config.smplx_folder,
                    save_folder=os.path.join(save_folder,name, 'rendering'),
                    scene_mesh=static_scene,
                    pkl = pkl,
                    num_betas=10,
                )

                frame2video(
                    path=os.path.join(save_folder,name, 'rendering/%03d.png'),
                    video=os.path.join(save_folder,name, 'motion.mp4'),
                    start=0,
                )



                for j in range(N):
                    pred_n = pred_expmap[i,:,j]
                    target_n = targ_expmap[i,:,j]

                    path_name = os.path.join(save_folder,name,str(j))
                    if not os.path.exists(path_name):
                        os.mkdir(path_name)
                    plt.plot(pred_n.cpu().detach().numpy(),label = "prediction")
                    plt.plot(target_n.cpu().detach().numpy(),label = "target")
                    plt.xlabel('Frame')
                    plt.ylabel("SDF")
                    plt.legend()
                    
                    plt.savefig(os.path.join(path_name,"SDF.png"))
                    plt.close()
                    render_motion_point_in_scene(
                    smplx_folder= config.smplx_folder,
                    save_folder=os.path.join(save_folder,name,str(j)),
                    scene_mesh=static_scene,
                    pkl = pkl,
                    num_betas=10,
                    marker = marker_indic_62[j]
                )


    def visualize_prediction(self):
        print("load model...")
        start_epoch = self._load_state_dict()
        print("load done")
        ind = 0
        scene_list = {}
        for data in self.dataloader['val']:
            [scene_id, scene_radius, scene_trans, motion_transformation, scene_sdf,  trans, orient, betas, pose_body, pose_hand, motion_sdf,scene_wsdf] = data
            
            # print(scene_trans.shape,motion_transformation.shape,scene_sdf.shape,trans.shape,orient.shape,pose_body.shape,pose_hand.shape)
            hlen = self.config.history_len
            history_trans = trans[:,:hlen,:]
            history_orient = orient[:,:hlen,:]
            history_pose_body = pose_body[:,:hlen,:]
            history_pose_hand = pose_hand[:,:hlen,:]
            future_trans = trans[:,hlen:,:]
            future_orient = orient[:,hlen:,:]
            future_pose_body = pose_body[:,hlen:,:]
            future_pose_hand = pose_hand[:,hlen:,:]
            future_motion_sdf = motion_sdf[:,hlen:,:]
            ## forward
            # future_motion_sdf = self._get_motion_sdf(future_trans,future_orient,future_pose_body,future_pose_hand,betas,scene_sdf,scene_radius, True)
            # print(future_motion_sdf.shape)
            # motionsdf_time = time.time()
            # print("get motion sdf cost:", motionsdf_time - start)
            [pred_trans, pred_orient, pred_pose_body, pred_pose_hand] = self._forward(
                history_trans, history_orient, history_pose_body, history_pose_hand, scene_sdf, future_motion_sdf
            )

            B = pred_trans.shape[0]
            S = pred_trans.shape[1]
            for b in range(B):
                sid = scene_id[b]
                # if sid in scene_list.keys():
                #     continue
                # else:
                #     scene_list[sid] = 1

                print('drawing',sid)

            

                mt = motion_transformation[b,:].cpu().detach().numpy()
                pred_t = pred_trans[b,:,:]
                pred_o = pred_orient[b,:,:]
                pred_p =pred_pose_body[b,:,:]
                pred_pd = pred_pose_hand[b,:,:]
                pred_t1 = pred_t.cpu().detach().numpy()
                pred_o1 = pred_o.cpu().detach().numpy()
                pred_pb1 = pred_p.cpu().detach().numpy()
                pred_pd1 = pred_pd.cpu().detach().numpy()
                b1 = betas[b,:].cpu().detach().numpy()

                pred_o1 = GeometryTransformer.convert_to_3D_rot(pred_o.detach()).cpu().numpy()
                body_vertices_pred, body_faces_pred, _ = SMPLX_Util.get_body_vertices_sequence(
                    config.smplx_folder, 
                    (pred_t1+mt, pred_o1, b1, pred_pb1, pred_pd1),
                    num_betas=10
                    )
                
                gt_t = future_trans[b,:,:].cpu().detach().numpy()
                gt_o = future_orient[b,:,:].cpu().detach().numpy()
                gt_pb = future_pose_body[b,:,:].cpu().detach().numpy()
                gt_pd = future_pose_hand[b,:,:].cpu().detach().numpy()
                gt_o = GeometryTransformer.convert_to_3D_rot(future_orient[b,:,:].detach()).cpu().numpy()
                body_vertices_gt, body_faces_gt, _ = SMPLX_Util.get_body_vertices_sequence(
                    config.smplx_folder, 
                    (gt_t+mt, gt_o, b1, gt_pb, gt_pd),
                    num_betas=10
                    )


                his_t = history_trans[b,:,:].cpu().detach().numpy()
                his_o = history_orient[b,:,:].cpu().detach().numpy()
                his_pb = history_pose_body[b,:,:].cpu().detach().numpy()
                his_pd = history_pose_hand[b,:,:].cpu().detach().numpy()
                his_o = GeometryTransformer.convert_to_3D_rot(history_orient[b,:,:].detach()).cpu().numpy()
                body_vertices_his, body_faces_his, _ = SMPLX_Util.get_body_vertices_sequence(
                    config.smplx_folder, 
                    (his_t+mt, his_o, b1, his_pb, his_pd),
                    num_betas=10
                    )
                sid = scene_id[b]

                scene_path =  "/hdd/Pointnet2.ScanNet/possion_reconstruction/scene/{}_00/{}_00_vh_clean_2.ply".format(sid,sid)
                static_scene = trimesh.load(scene_path, process=False)
                # Sc =trimesh.Scene()
                # for i in range(0, len(body_vertices_gt), 5):
                #     Sc.add_geometry(
                #         trimesh.Trimesh(vertices=body_vertices_gt[i], faces=body_faces_gt)
                #     )
                # # print("length of body verts", body_verts1.shape, body_faces1.shape)
                # # # print(scene_radius,body_verts1)
                # # for i in range(0, len(body_verts1), 2):
                # #     Sc.add_geometry(
                # #         trimesh.Trimesh(vertices=body_verts1[i]+mt[b,:], faces=body_faces1)
                # #     )
                # # Sc.add_geometry(
                # #     trimesh.Trimesh(vertices=body_verts1[0]+mt[b,:], faces=body_faces1)
                # # )
                # Sc.add_geometry(static_scene)
                # Sc.show()

                
                # Sc1 =trimesh.Scene()
                # for i in range(0, len(body_vertices_pred), 5):
                #     Sc1.add_geometry(
                #         trimesh.Trimesh(vertices=body_vertices_pred[i], faces=body_faces_pred)
                #     )
                # # print("length of body verts", body_verts1.shape, body_faces1.shape)
                # # # print(scene_radius,body_verts1)
                # # for i in range(0, len(body_verts1), 2):
                # #     Sc.add_geometry(
                # #         trimesh.Trimesh(vertices=body_verts1[i]+mt[b,:], faces=body_faces1)
                # #     )
                # # Sc.add_geometry(
                # #     trimesh.Trimesh(vertices=body_verts1[0]+mt[b,:], faces=body_faces1)
                # # )
                # Sc1.add_geometry(static_scene)
                # Sc1.show()

                pkl1 = (gt_t+mt, gt_o, b1, gt_pb, gt_pd)
                pkl2 = (pred_t1+mt, pred_o1, b1, pred_pb1, pred_pd1)
                pkl3 = (his_t+mt,his_o,b1,his_pb,his_pd)
                name = sid+str(ind)
                ind+=1
                save_folder = config.save_vis_folder
                render_reconstructed_motion_in_scene(
                smplx_folder=config.smplx_folder,
                save_folder=os.path.join(save_folder,name, 'rendering'),
                pkl_rec=pkl2,
                scene_mesh=static_scene,
                pkl_gt =pkl1,
                pkl_his = pkl3,
                num_betas=self.config.num_betas
                )


                frame2video(
                    path=os.path.join(save_folder,name, 'rendering/%03d.png'),
                    video=os.path.join(save_folder,name, 'motion.mp4'),
                    start=0,
                )




    def check_data(self):
        print('-' * 20, 'check data', '-' * 20)
        for data in self.dataloader['train']:
            [scene_id, scene_radius, scene_trans, motion_transformation, scene_sdf,  trans, orient, betas, pose_body, pose_hand, motion_sdf,scene_wsdf] = data
            
            hlen = self.config.history_len
            
            future_trans = trans[:,hlen:,:]
            future_orient = orient[:,hlen:,:]
            future_pose_body = pose_body[:,hlen:,:]
            future_pose_hand = pose_hand[:,hlen:,:]
            B, S, _ = future_trans.shape
            mt = motion_transformation.cpu().detach().numpy()
            future_orient = GeometryTransformer.convert_to_3D_rot(future_orient.reshape(-1, 6)).reshape(B,S,3)
        
            motion_transformation = motion_transformation.unsqueeze(1).repeat(1,30,1)
            motion_sdf_v3 = self.get_sdf_grid_batch(future_trans+motion_transformation,future_orient,future_pose_body,future_pose_hand,betas,scene_wsdf,scene_radius)
            
            
            for b in range(2,B):
                for i in range(0,len(marker_indic),10):
                    # print(marker_indic)
                    ind = i
                    print('check', b,ind )
                    plt.plot(motion_sdf_v3[b,:,ind].cpu().detach().numpy(),label = "V3")
                    plt.plot(motion_sdf[b,hlen:,ind].cpu().detach().numpy(),label = "GT")
                    plt.xlabel('Frame')
                    plt.ylabel("SDF")
                    plt.legend()
                    plt.show()
                

    



    def _convert_compute_smplx_to_render_smplx(self, smplx_tensor_tuple):
        trans1, orient1, betas1, pose_body1, pose_hand1 = smplx_tensor_tuple
        trans1 = trans1.detach().cpu().numpy()
        orient1 = GeometryTransformer.convert_to_3D_rot(orient1.detach()).cpu().numpy()
        betas1 = betas1.detach().cpu().numpy()
        pose_body1 = pose_body1.detach().cpu().numpy()
        pose_hand1 = pose_hand1.detach().cpu().numpy()

        return (trans1, orient1, betas1, pose_body1, pose_hand1)

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

    
    def _epoch_report_val(self, epoch_id: int):
        # print(overlap_maker_indic,"overlap")
        Console.log("epoch [{}/{}] done...".format(epoch_id+1, self.config.num_epoch))


        epoch_report_str = EPOCH_VAL_REPORT_TEMPLATE.format(
            # train_rec_sdf_loss=round(np.mean(self.log['train'][epoch_id]['smooth_loss1']), 5),
            val_sdf_loss=round(np.mean(self.log['val'][epoch_id]['sdf_loss']), 5),
            val_hsdf_loss=round(np.mean(self.log['val'][epoch_id]['hsdf_loss']), 5),
            # val_rec_sdf_loss=round(np.mean(self.log['val'][epoch_id]['smooth_loss1']), 5),
        )
        Console.log(epoch_report_str)




    
    def _epoch_report(self, epoch_id: int):
        Console.log("epoch [{}/{}] done...".format(epoch_id+1, self.config.num_epoch))


        epoch_report_str = EPOCH_REPORT_TEMPLATE.format(
            train_sdf_loss=round(np.mean(self.log['train'][epoch_id]['sdf_loss']), 5),
            train_hsdf_loss=round(np.mean(self.log['train'][epoch_id]['hsdf_loss']), 5),
            val_sdf_loss=round(np.mean(self.log['val'][epoch_id]['sdf_loss']), 5),
            val_hsdf_loss=round(np.mean(self.log['val'][epoch_id]['hsdf_loss']), 5)
           
            # val_rec_sdf_loss=round(np.mean(self.log['val'][epoch_id]['smooth_loss1']), 5),
        )
        Console.log(epoch_report_str)
    
    def _best_report(self):
        Console.log("training completed...")

        best_report_str = BEST_REPORT_TEMPLATE.format(
            best_epoch=self.best['epoch'],
            best_total_loss=self.best['sdf_loss']
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
    
    