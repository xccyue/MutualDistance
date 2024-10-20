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
from model.basemodel import MotionGRU, SceneNet, SDFNet, GCN_H, GCN_S, MotionGRU_S1
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
# import kaolin
from utils.data_utils import get_dct_matrix
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
"""
EPOCH_VAL_REPORT_TEMPLATE = """
----------------------summary----------------------
Stage1
[val] val_pred_sdf_loss: {val_pred_sdf_loss}
[val] val_pred_hsdf_loss: {val_pred_hsdf_loss}
Stage2
[val] val_total_loss: {val_total_loss}
[val] val_rec_loss: {val_rec_loss}
[val] val_rec_trans_loss: {val_rec_trans_loss}
[val] val_rec_orient_loss: {val_rec_orient_loss}
[val] val_rec_body_pose_loss: {val_rec_body_pose_loss}
[val] val_rec_hand_pose_loss: {val_rec_hand_pose_loss}
[val] val_sdf_loss: {val_rec_sdf_loss}
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
        self.scene_net_s2 = SceneNet(self.config).to(self.config.device)
        # self.motion_net = MotionNet(self.config).to(self.config.device)
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
        # self.scene_points = torch.FloatTensor(data)

        
        self.scene_points = torch.FloatTensor(data).unsqueeze(0).repeat(config.batch_size*config.motion_len,1,1).cuda()
        self.scene_points1 = torch.FloatTensor(data).unsqueeze(0).repeat(config.batch_size*config.future_len,1,1).cuda()



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
        self.scheduler = get_scheduler(self.optimizer_h, policy='step', decay_step=30)

        # log
        # contains all necessary info for all phases
        self.log = {phase: {} for phase in ["train", "val"]}
        self.dump_keys = ['loss', 'rec_loss',  'rec_trans_loss', 'rec_orient_loss', 'rec_body_pose_loss', 'rec_hand_pose_loss', 'rec_sdf_loss','smooth_loss1', 'rec_hsdf_loss',' pred_hsdf_loss', 'pred_msdf_loss']

        self.best = {
            'epoch': 0,
            'loss': float("inf"),
            'rec_loss': float("inf"),
            'rec_sdf_loss': float("inf"),
            'rec_trans_loss' : float("inf"),
            "rec_orient_loss" : float("inf"),
            "rec_body_pose_loss" : float("inf"),
            "rec_hand_pose_loss" : float("inf")
        }

        # report model size
        # self._report_model_size()

        # vp, ps = load_model(config.vposer_folder, model_code=VPoser,
        #                     remove_words_in_model_weights='vp_model.',
        #                     disable_grad=True)
        # self.vposer = vp.to(self.config.device)
        print("smplx initialize", self.config.batch_size * self.config.future_len)
        self.smplx_model = smplx.create(config.smplx_folder, model_type='smplx',
            gender='neutral', ext='npz',
            num_betas=self.config.num_betas,
            use_pca=False,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=True,
            batch_size=self.config.batch_size * self.config.future_len,
        ).to(self.config.device)
        self.smplx_model_h = smplx.create(config.smplx_folder, model_type='smplx',
            gender='neutral', ext='npz',
            num_betas=self.config.num_betas,
            use_pca=False,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=True,
            batch_size=self.config.batch_size * self.config.motion_len,
        ).to(self.config.device)


    def _report_model_size(self):
        # sum_scene = sum([param.nelement() for param in self.cond_net.scene_model.parameters()])
        # sum_text = sum([param.nelement() for param in self.cond_net.bert_model.parameters()])
        # sum_cond_net = sum([param.nelement() for param in self.cond_net.parameters()])
        # sum_cond_rest = sum_cond_net - sum_text - sum_scene
        
        # sum_base_model = sum([param.nelement() for param in self.base_model.parameters()])
        # sum_encoder = sum([param.nelement() for param in self.base_model.encoder.parameters()])
        # sum_decoder = sum([param.nelement() for param in self.base_model.decoder.parameters()])

        sum_scene = sum([param.nelement() for param in self.scene_net.parameters()])
        # sum_motion = sum([param.nelement() for param in self.motion_net.parameters()])
        sum_sdf = sum([param.nelement() for param in self.sdf_net.parameters()])
        sum_encoder = sum_scene + sum_sdf
        sum_motion_decoder = sum([param.nelement() for param in self.motion_decoder.parameters()])
        sum_decoder = sum_motion_decoder

        # Console.log(
        #     'sum_scene: ({}) + sum motion: ({}) + sum sdf: ({}) = sum encoder: ({}), sum motion decoder: ({}) = sum_decoder: ({})'.format(
        #         sum_scene, sum_motion, sum_sdf, sum_encoder, sum_motion_decoder, sum_decoder,
        #     )
        # )
        Console.log(
            ' sum_scene: ({}) + sum_sdf: ({})= sum encoder: ({}), sum motion decoder: ({}) = sum_decoder: ({})'.format(
              sum_scene,sum_sdf, sum_encoder, sum_motion_decoder, sum_decoder,
            )
        )
        Console.log('all parameters: {}'.format(sum_decoder+sum_encoder))
    
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
            # ## load cond net weight
            # cond_net_dict = self.cond_net.state_dict()
            # cond_net_dict.update(state_dict['cond_net_state_dict'])
            # self.cond_net.load_state_dict(cond_net_dict)
            # ## load cvae model weight
            # self.base_model.load_state_dict(state_dict['base_model_state_dict'])
            print(state_dict.keys())
            self.motion_decoder.load_state_dict(state_dict['motion_decoder_state_dict'])
            self.sdf_net.load_state_dict(state_dict['sdf_net_state_dict'])
            self.scene_net_s2.load_state_dict(state_dict['scene_net_state_dict'])
            # self.optimizer_h.load_state_dict(state_dict=['optimizer_h_state_dict'])
            # self.scheduler.load_state_dict(state_dict=['scheduler_state_dict'])
            Console.log('Load checkpoint: {}. start from {}'.format(ckp_file, state_dict['epoch']))






            if os.path.isdir(self.config.resume_model_s1):
                ckp_file = os.path.join(self.config.resume_model_s1, 'epoch40.pth')
            elif os.path.isfile(self.config.resume_model_s1):
                ckp_file = self.config.resume_model_s1
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
            self.scene_net_s1.load_state_dict(state_dict['scene_net_state_dict'])
            # self.optimizer_h.load_state_dict(state_dict=['optimizer_h_state_dict'])
            # self.scheduler.load_state_dict(state_dict=['scheduler_state_dict'])
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

    def _forward(
        self, history_trans, history_orient, history_pose_body, history_pose_hand, scene_sdf, sdf_in,betas, hsdf_in
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
        B, S, _ = history_trans.shape
        trans_shape = history_trans.shape[-1]
        orient_shape = history_orient.shape[-1]
        pose_body_shape = history_pose_body.shape[-1]
        pose_hand_shape = history_pose_hand.shape[-1]
        # print(trans_shape,orient_shape,pose_body_shape,pose_hand_shape)
        # B S 162
        history_motion = torch.cat((history_trans,history_orient, history_pose_body, history_pose_hand),dim=-1)
        # print("************")
        # B S 162
        # print(history_motion[:,-1,:3])
        history_motion = history_motion.permute(1,0,2)
        scene_sdf = scene_sdf.unsqueeze(1)

        fs = self.scene_net_s1(scene_sdf)
        # fs B 128
        fh = self.motion_encoder(history_motion)
        # fh B 128
        # sdf_in B N L
        # beta B 10
        pred_sdf = self.sdf_gcn(sdf_in,fs,fh,betas)
        pred_hsdf = self.hsdf_gcn(hsdf_in,fs,fh,betas)



        T = 45
        N = sdf_in.shape[1]
        dct_n = self.dct_n
        _, idct_m = get_dct_matrix(T)
        idct_m = torch.FloatTensor(idct_m).cuda()
        B = pred_sdf.shape[0]
        pred_sdf = pred_sdf.view(B,-1)
        # L B*N
        pred_t = pred_sdf.view(-1, dct_n).transpose(0,1)
        #L B*N -> T B*N -> B*N T -> B, N , T -> B T N
        pred_expmap = torch.matmul(idct_m[:, :dct_n], pred_t).transpose(0, 1).contiguous().view(-1, N,
                                                                                               T).transpose(1, 2)
        hl = self.config.history_len

        motion_sdf = pred_expmap



        T = 45
        N = hsdf_in.shape[1]
        dct_n = self.dct_n
        _, idct_m = get_dct_matrix(T)
        idct_m = torch.FloatTensor(idct_m).cuda()
        B = pred_hsdf.shape[0]
        pred_hsdf = pred_hsdf.view(B,-1)
        # L B*N
        pred_ht = pred_hsdf.view(-1, dct_n).transpose(0,1)
        #L B*N -> T B*N -> B*N T -> B, N , T -> B T N
        pred_hexpmap = torch.matmul(idct_m[:, :dct_n], pred_ht).transpose(0, 1).contiguous().view(-1, N,
                                                                                               T).transpose(1, 2)
        hl = self.config.history_len

        human_sdf = pred_hexpmap



        
        # print(history_motion[-1,:,:3])
        # print("----------------")
        # print("input shape", history_motion.shape) 25 64 162
        
        fs = self.scene_net_s2(scene_sdf)
        fsdf = self.sdf_net(motion_sdf)
        hsdf = human_sdf.permute(1,0,2)
        fsdf = fsdf.permute(1,0,2)
        output = self.motion_decoder(history_motion, fs, fsdf, betas, hsdf)
        output = output.permute(1,0,2)
        
        # print("output",output.shape)
        pred_trans = output[:,:,:trans_shape]
        pred_orient = output[:,:,trans_shape:trans_shape+orient_shape]
        pred_body_pose = output[:,:,trans_shape+orient_shape:trans_shape+orient_shape+pose_body_shape]
        pred_hand_pose = output[:,:,trans_shape+orient_shape+pose_body_shape:]
        return pred_trans, pred_orient, pred_body_pose, pred_hand_pose, motion_sdf, human_sdf


    
    def get_sdf_grid_batch(self, pred_trans,orient,pose_body,pose_hand,betas, scene_radius, scene_sdf1, scene_index1):
        B = pred_trans.shape[0]
        S = pred_trans.shape[1]

 
        orient = orient.reshape(B*S,orient.shape[-1])
        pose_body = pose_body.reshape(B*S,pose_body.shape[-1])
        pose_hand = pose_hand.reshape(B*S,pose_hand.shape[-1])
        betas_seq = betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1)

        pred_trans = pred_trans.reshape(B*S,pred_trans.shape[-1])
        verts, _ = self._get_body_vertices(
            pred_trans,
            orient,
            betas_seq,
            pose_body,
            pose_hand
        )
        verts = verts.reshape(B,S,-1,3)
        verts = verts[:,:,marker_indic,:]
        B,S,N,_ = verts.shape
        lo1 = verts.reshape(B,S*N,3)
        scene_index1 = scene_index1.unsqueeze(1)
        lo1 =2*((lo1 - scene_index1[:,:,:3])/(scene_index1[:,:,3:] - scene_index1[:,:,:3]) - 0.5)
        sdf1 = F.grid_sample(scene_sdf1[:,None],lo1[:,None,None,:,[2,1,0]], align_corners=True)
        sdf1 = sdf1.reshape(B,S,N)

        return sdf1


            
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
        path_error = torch.sqrt(((pred_trans - future_trans) ** 2).sum(-1))
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
        joints_gt = joints_gt.reshape(B,S,-1,3)
        joints_rec = joints_rec.reshape(B,S,-1,3)
        pose_error = torch.sqrt(((joints_gt - joints_rec) ** 2).sum(-1)).mean(dim=(2))
        # print(pose_error.shape)



        return rec_joints_error, rec_vertex_error, path_error, pose_error
    


    def _cal_loss(self, future_trans, future_orient,future_pose_body,future_pose_hand,
                pred_trans, pred_orient, pred_pose_body, pred_pose_hand, betas,
                scene_radius, future_motion_sdf, motion_transformation, scene_sdf, scene_index, future_hsdf):
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
        

        B, S, _ = future_trans.shape
        

        ## rec loss, smplx parameters
        # print("future trans", future_trans.shape)
        rec_trans_loss = F.l1_loss(pred_trans,future_trans)
        rec_orient_loss = F.l1_loss(pred_orient,future_orient)
        rec_body_pose_loss = F.l1_loss(pred_pose_body,future_pose_body)
        rec_hand_pose_loss = F.l1_loss(pred_pose_hand,future_pose_hand)
        gt_hsdf = future_hsdf
        pred_hsdf = self._get_hsdf_future(pred_trans,pred_orient, betas, pred_pose_body, pred_pose_hand, scene_index)
        rec_hsdf_loss = F.l1_loss(pred_hsdf, gt_hsdf)
        ## 1. motion_sdf_loss
        gt_motion_sdf = future_motion_sdf
        motion_transformation = motion_transformation.unsqueeze(1).repeat(1,S,1)
        pred_orient = GeometryTransformer.convert_to_3D_rot(pred_orient.reshape(-1, 6)).reshape(B,S,3)
        pred_motion_sdf = self.get_sdf_grid_batch(pred_trans,pred_orient,pred_pose_body,pred_pose_hand,betas,scene_radius, scene_sdf, scene_index)
        rec_sdf_loss = F.l1_loss(pred_motion_sdf, gt_motion_sdf)



        
        smooth_loss1 = 0

        
        return  rec_trans_loss, rec_orient_loss, rec_body_pose_loss,rec_hand_pose_loss, rec_sdf_loss, smooth_loss1, rec_hsdf_loss


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

    def _get_hsdf(self, trans,orient, betas, pose_body, pose_hand, scene_index):
        B = trans.shape[0]
        S = trans.shape[1]

        orient = GeometryTransformer.convert_to_3D_rot(orient.reshape(-1, 6)).reshape(B,S,3)
        orient = orient.reshape(B*S,orient.shape[-1])
        pose_body = pose_body.reshape(B*S,pose_body.shape[-1])
        pose_hand = pose_hand.reshape(B*S,pose_hand.shape[-1])
        betas_seq = betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1)

        trans = trans.reshape(B*S,trans.shape[-1])
        verts, joints, faces = self._get_body_vertices_h(
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
        # print(self.scene_points.shape, verts.shape, scene_index.shape)
        distance = torch.cdist(self.scene_points,verts)
        distance = torch.min(distance,dim=2)[0]
        distance = distance.reshape(B,S,-1)
        # print(torch.max(torch.abs(distance1 - distance)))

        return distance


    def _train(self, train_dataloader: DataLoader, epoch_id: int):
        phase = 'train'
        self.log[phase][epoch_id] = defaultdict(list)
        dct_n =45
        dct_m_in, _ = get_dct_matrix(dct_n)
        pad_idx = np.repeat([15-1],30)
        i_idx = np.append(np.arange(0,15),pad_idx)
        dct_m_in = torch.tensor(dct_m_in).cuda()
        B = config.batch_size
        dct_m_in = dct_m_in.unsqueeze(0).repeat(B,1,1).float()

        for data in tqdm(train_dataloader):
            start = time.time()

            ## unpack data
            [scene_id, scene_radius,  scene_trans, motion_transformation, scene_sdf,  trans, orient, betas, pose_body, pose_hand,motion_sdf, scene_index, in_sdf, out_sdf, motion_hsdf, in_hsdf] = data
          
            
       
            # print(scene_trans.shape,motion_transformation.shape,scene_sdf.shape,trans.shape,orient.shape,pose_body.shape,pose_hand.shape)
            hlen = self.config.history_len
            # print("hlen", hlen) 25
            history_trans = trans[:,:hlen,:]
            history_orient = orient[:,:hlen,:]
            history_pose_body = pose_body[:,:hlen,:]
            history_pose_hand = pose_hand[:,:hlen,:]
            # print("################")
            # print(history_trans[:,-1,:])
            # print("his orient",history_orient.shape) 64 25 6
            future_trans = trans[:,hlen:,:]
            future_orient = orient[:,hlen:,:]
            future_pose_body = pose_body[:,hlen:,:]
            future_pose_hand = pose_hand[:,hlen:,:]
            # print("future orient", future_orient.shape) 64 25 6
            ## forward
            future_motion_sdf = motion_sdf[:,hlen:,:]
            motion_hsdf = self._get_hsdf(trans,orient,betas,pose_body,pose_hand,scene_index)


            hsdf = motion_hsdf
            future_hsdf = motion_hsdf[:,hlen:,:]
            hsdf_in = torch.bmm(dct_m_in[:,:dct_n,:],motion_hsdf[:,i_idx,:])
            hsdf_in = hsdf_in.transpose(1,2)
            # print("-----------------")
            # future_motion_sdf = self._get_motion_sdf(future_trans,future_orient,future_pose_body,future_pose_hand,betas,scene_sdf,scene_radius, True)
            # print(future_motion_sdf.shape)
            motion_sdf_time = time.time()
            [pred_trans, pred_orient, pred_pose_body, pred_pose_hand, pred_msdf, pred_hsdf] = self._forward(
                history_trans, history_orient, history_pose_body, history_pose_hand, scene_sdf, in_sdf,betas, hsdf_in
            )

            pred_hsdf_loss = F.l1_loss(pred_hsdf[:,hlen:,:], hsdf[:,hlen:,:])
            pred_msdf_loss = F.l1_loss(pred_msdf[:,hlen:,:], motion_sdf[:,hlen:,:])
            
           
            # print(pred_trans.shape)
            forward_time = time.time()
            [ rec_trans_loss, rec_orient_loss, rec_body_pose_loss, rec_hand_pose_loss, rec_sdf_loss, smooth_loss1, rec_hsdf_loss] = self._cal_loss(
                future_trans, future_orient,future_pose_body,future_pose_hand,
                pred_trans, pred_orient, pred_pose_body, pred_pose_hand, betas,
                scene_radius, future_motion_sdf, motion_transformation, scene_sdf, scene_index, future_hsdf
            )
            cal_loss_time = time.time()
            # rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
            #             self.config.weight_loss_rec * rec_orient_loss + \
            #             self.config.weight_loss_rec_body_pose * rec_body_pose_loss + \
            #             self.config.weight_loss_rec_hand_pose * rec_hand_pose_loss
            # loss = self.config.weight_loss_sdf * rec_sdf_loss + \
            #         rec_loss
            rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
                        self.config.weight_loss_rec * rec_orient_loss + \
                        self.config.weight_loss_rec_body_pose * rec_body_pose_loss + \
                        self.config.weight_loss_rec_hand_pose * rec_hand_pose_loss
            loss = self.config.weight_loss_sdf * rec_sdf_loss + \
            rec_loss + self.config.weight_loss_sdf * rec_hsdf_loss + self.config.weight_loss_pred_hsdf * pred_hsdf_loss + self.config.weight_loss_pred_msdf * pred_msdf_loss

            ## backward
            self.optimizer_h.zero_grad()
            loss.backward()
            self.optimizer_h.step()
            back_time =time.time()

            # print("motion sdf", motion_sdf_time- start, "forward", forward_time-motion_sdf_time,"loss",cal_loss_time-forward_time,"back",back_time-cal_loss_time)
            ## record log
            iter_time = time.time() - start
            self.log[phase][epoch_id]['loss'].append(loss.item())
            self.log[phase][epoch_id]['rec_loss'].append(rec_loss.item())
            self.log[phase][epoch_id]['rec_trans_loss'].append(rec_trans_loss.item())
            self.log[phase][epoch_id]['rec_orient_loss'].append(rec_orient_loss.item())
            self.log[phase][epoch_id]['rec_body_pose_loss'].append(rec_body_pose_loss.item())
            self.log[phase][epoch_id]['rec_hand_pose_loss'].append(rec_hand_pose_loss.item())
            self.log[phase][epoch_id]['rec_sdf_loss'].append(rec_sdf_loss.item())
            self.log[phase][epoch_id]['rec_hsdf_loss'].append(rec_hsdf_loss.item())
            self.log[phase][epoch_id]['pred_hsdf_loss'].append(pred_hsdf_loss.item())
            self.log[phase][epoch_id]['pred_msdf_loss'].append(pred_msdf_loss.item())
            # self.log[phase][epoch_id]['smooth_loss1'].append(smooth_loss1.item())
        self.scheduler.step()
        my_lr = self.scheduler.optimizer.param_groups[0]['lr']
        print('lr',my_lr)
    
    def _val(self, val_dataloader: DataLoader, epoch_id: int):
        phase = 'val'
        self.log[phase][epoch_id] = defaultdict(list)
        result = {}
        dct_n =45
        dct_m_in, _ = get_dct_matrix(dct_n)
        pad_idx = np.repeat([15-1],30)
        i_idx = np.append(np.arange(0,15),pad_idx)
        dct_m_in = torch.tensor(dct_m_in).cuda()
        B = config.batch_size
        dct_m_in = dct_m_in.unsqueeze(0).repeat(B,1,1).float()

        for data in tqdm(val_dataloader):
            start = time.time()

            ## unpack data
            [scene_id, scene_radius,  scene_trans, motion_transformation, scene_sdf,  trans, orient, betas, pose_body, pose_hand,motion_sdf, scene_index, in_sdf, out_sdf, motion_hsdf, in_hsdf] = data
          
            
       
            # print(scene_trans.shape,motion_transformation.shape,scene_sdf.shape,trans.shape,orient.shape,pose_body.shape,pose_hand.shape)
            hlen = self.config.history_len
            # print("hlen", hlen) 25
            history_trans = trans[:,:hlen,:]
            history_orient = orient[:,:hlen,:]
            history_pose_body = pose_body[:,:hlen,:]
            history_pose_hand = pose_hand[:,:hlen,:]
            # print("################")
            # print(history_trans[:,-1,:])
            # print("his orient",history_orient.shape) 64 25 6
            future_trans = trans[:,hlen:,:]
            future_orient = orient[:,hlen:,:]
            future_pose_body = pose_body[:,hlen:,:]
            future_pose_hand = pose_hand[:,hlen:,:]
            # print("future orient", future_orient.shape) 64 25 6
            ## forward
            future_motion_sdf = motion_sdf[:,hlen:,:]
            motion_hsdf = self._get_hsdf(trans,orient,betas,pose_body,pose_hand,scene_index)


            hsdf = motion_hsdf
            future_hsdf = motion_hsdf[:,hlen:,:]
            hsdf_in = torch.bmm(dct_m_in[:,:dct_n,:],motion_hsdf[:,i_idx,:])
            hsdf_in = hsdf_in.transpose(1,2)
            # print("-----------------")
            # future_motion_sdf = self._get_motion_sdf(future_trans,future_orient,future_pose_body,future_pose_hand,betas,scene_sdf,scene_radius, True)
            # print(future_motion_sdf.shape)
            motion_sdf_time = time.time()
            [pred_trans, pred_orient, pred_pose_body, pred_pose_hand, pred_msdf, pred_hsdf] = self._forward(
                history_trans, history_orient, history_pose_body, history_pose_hand, scene_sdf, in_sdf,betas, hsdf_in
            )

            # print(pred_msdf.shape, pred_hsdf.shape)

            pred_hsdf_loss = F.l1_loss(pred_hsdf[:,hlen:,:],hsdf[:,hlen:,:])
            pred_msdf_loss = F.l1_loss(pred_msdf[:,hlen:,:],motion_sdf[:,hlen:,:])
            # print(pred_trans.shape)
            forward_time = time.time()
            [ rec_trans_loss, rec_orient_loss, rec_body_pose_loss, rec_hand_pose_loss, rec_sdf_loss, smooth_loss1, rec_hsdf_loss] = self._cal_loss(
                future_trans, future_orient,future_pose_body,future_pose_hand,
                pred_trans, pred_orient, pred_pose_body, pred_pose_hand, betas,
                scene_radius, future_motion_sdf, motion_transformation, scene_sdf, scene_index, future_hsdf
            )
            cal_loss_time = time.time()
            # rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
            #             self.config.weight_loss_rec * rec_orient_loss + \
            #             self.config.weight_loss_rec_body_pose * rec_body_pose_loss + \
            #             self.config.weight_loss_rec_hand_pose * rec_hand_pose_loss
            # loss = self.config.weight_loss_sdf * rec_sdf_loss + \
            #         rec_loss
            rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
                        self.config.weight_loss_rec * rec_orient_loss + \
                        self.config.weight_loss_rec_body_pose * rec_body_pose_loss + \
                        self.config.weight_loss_rec_hand_pose * rec_hand_pose_loss
            loss = self.config.weight_loss_sdf * rec_sdf_loss + \
            rec_loss + self.config.weight_loss_hsdf * rec_hsdf_loss + self.config.weight_loss_pred_hsdf * pred_hsdf_loss + self.config.weight_loss_pred_msdf * pred_msdf_loss

            rec_joints_error, rec_vertex_error, path_error, pose_error = self._compute_rec_error(future_trans, future_orient,future_pose_body,future_pose_hand,
                pred_trans, pred_orient, pred_pose_body, pred_pose_hand, betas,
               scene_radius, future_motion_sdf, motion_transformation)
            scene_id, scene_radius,  scene_trans, motion_transformation, scene_sdf,  trans, orient, betas, pose_body, pose_hand,motion_sdf, scene_index, in_sdf, out_sdf, motion_hsdf, in_hsdf
            B = trans.shape[0]

            # for b in range(B):
            #     da = {}
            #     da['scene_trans'] = scene_trans[b].detach().cpu().numpy()
            #     da['motion_transformation'] = motion_transformation[b].detach().cpu().numpy()
            #     da['trans'] = trans[b].detach().cpu().numpy()
            #     da['orient'] = orient[b].detach().cpu().numpy()
            #     da['betas'] = betas[b].detach().cpu().numpy()
            #     da['pose_body'] = pose_body[b].detach().cpu().numpy()
            #     da['pose_hand'] = pose_hand[b].detach().cpu().numpy()
            #     da['motion_sdf'] = motion_sdf[b].detach().cpu().numpy()
            #     da['motion_hsdf'] = motion_hsdf[b].detach().cpu().numpy()
            #     da['path_error'] = path_error[b].detach().cpu().numpy()
            #     da['pose_error'] = pose_error[b].detach().cpu().numpy()
          
            #     da['pred_trans'] = pred_trans[b].detach().cpu().numpy()
            #     da['pred_orient'] = pred_orient[b].detach().cpu().numpy()
            #     da['pred_pose_body'] = pred_pose_body[b].detach().cpu().numpy()
            #     da['pred_pose_hand'] = pred_pose_hand[b].detach().cpu().numpy()
            #     da['pred_msdf'] = pred_msdf[b].detach().cpu().numpy()
            #     da['pred_hsdf'] = pred_hsdf[b].detach().cpu().numpy()
            #     result[scene_id[b]] = da
           
            path_error = path_error.mean(dim = 0)
            pose_error = pose_error.mean(dim = 0)



            # print("motion sdf", motion_sdf_time- start, "forward", forward_time-motion_sdf_time,"loss",cal_loss_time-forward_time,"back",back_time-cal_loss_time)
            ## record log
            iter_time = time.time() - start
            self.log[phase][epoch_id]['loss'].append(loss.item())
            self.log[phase][epoch_id]['rec_loss'].append(rec_loss.item())
            self.log[phase][epoch_id]['rec_trans_loss'].append(rec_trans_loss.item())
            self.log[phase][epoch_id]['rec_orient_loss'].append(rec_orient_loss.item())
            self.log[phase][epoch_id]['rec_body_pose_loss'].append(rec_body_pose_loss.item())
            self.log[phase][epoch_id]['rec_hand_pose_loss'].append(rec_hand_pose_loss.item())
            self.log[phase][epoch_id]['rec_sdf_loss'].append(rec_sdf_loss.item())
            self.log[phase][epoch_id]['rec_hsdf_loss'].append(rec_hsdf_loss.item())
            self.log[phase][epoch_id]['pred_hsdf_loss'].append(pred_hsdf_loss.item())
            self.log[phase][epoch_id]['pred_msdf_loss'].append(pred_msdf_loss.item())
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


            ## report log
            self._epoch_report(epoch_id)
            self._dump_log(epoch_id)

        # print best
        self._best_report()

        # save model
        Console.log("saving last models...\n")
        self._save_state_dict(epoch_id, 'model_last')

    
    def _epoch_report_val(self, epoch_id: int):
        Console.log("epoch [{}/{}] done...".format(epoch_id+1, self.config.num_epoch))
        # print(self.log['val'][epoch_id]['path_error'])
        # print(self.log['val'][epoch_id]['mpjpe'])
        # print(self.log['val'][epoch_id]['pose_error'])
        path_err = self.log['val'][epoch_id]['path_error']
        path_err = np.stack(path_err, axis = 0)
        print(path_err.shape)
        pose_err = self.log['val'][epoch_id]['pose_error']
        pose_err = np.stack(pose_err, axis = 0)
        
        print(pose_err.shape)
        path_err = np.mean(path_err, axis = 0).reshape(30)
        pose_err = np.mean(pose_err, axis = 0).reshape(30)
        print("0.5s, path error", path_err[14])
        print("1s, path_error", path_err[29])
        print("mean, path_error", path_err.mean())


        print("0.5s, pose error", pose_err[14])
        print("1s, pose_error", pose_err[29])
        print("mean, pose_error", pose_err.mean())

        print(path_err.shape, pose_err.shape)
        
        epoch_report_str = EPOCH_VAL_REPORT_TEMPLATE.format(
            val_total_loss=round(np.mean(self.log['val'][epoch_id]['loss']), 5),
            val_rec_loss=round(np.mean(self.log['val'][epoch_id]['rec_loss']), 5),
            val_rec_trans_loss=round(np.mean(self.log['val'][epoch_id]['rec_trans_loss']), 5),
            val_rec_orient_loss=round(np.mean(self.log['val'][epoch_id]['rec_orient_loss']), 5),
            val_rec_body_pose_loss=round(np.mean(self.log['val'][epoch_id]['rec_body_pose_loss']), 5),
            val_rec_hand_pose_loss=round(np.mean(self.log['val'][epoch_id]['rec_hand_pose_loss']), 5),
            val_rec_sdf_loss=round(np.mean(self.log['val'][epoch_id]['rec_sdf_loss']), 5),
            val_pred_sdf_loss=round(np.mean(self.log['val'][epoch_id]['pred_msdf_loss']), 5),
            val_pred_hsdf_loss=round(np.mean(self.log['val'][epoch_id]['pred_hsdf_loss']), 5),
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
            train_pred_sdf_loss=round(np.mean(self.log['train'][epoch_id]['pred_msdf_loss']), 5),
            train_pred_hsdf_loss=round(np.mean(self.log['train'][epoch_id]['pred_hsdf_loss']), 5),
            # train_rec_sdf_loss=round(np.mean(self.log['train'][epoch_id]['smooth_loss1']), 5),
           
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

        Ploter.write(dump_logs)
    
    