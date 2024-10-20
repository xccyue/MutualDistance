import glob
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
import utils.configuration as config
from utils.smplx_util import SMPLX_Util
from pyquaternion import Quaternion as Q
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer
from utils.utilities import Console
from utils.model_utils import GeometryTransformer
# from human_body_prior.tools.model_loader import load_model
# from human_body_prior.models.vposer_model import VPoser
# import smplx
from trimesh import transform_points
from natsort import natsorted
import utils.configuration as config
from utils.smplx_util import SMPLX_Util, marker_indic
from utils.data_utils import get_dct_matrix
############################################################
## Pose Dataset calss
############################################################
marker_indic = []


class MotionDataset_v1(Dataset):
    """ Motion Dataset for fetching a padded motion sequence data
    """
    def __init__(self, phase: str, motion_len: int=90, actions: list=[], num_betas: int=10, num_pca_comps: int=12, sdf_len: int=125):
        """ Init function, prepare train/val data
        """
        self.anno_folder = config.align_data_folder
        self.scene_folder = config.sdf_folder
        self.motion_folder = config.pure_motion_folder
        
        self.motion_length = config.history_len+config.future_len
        # print(self.scene_folder)
        self.phase = phase
        self.motion_len = motion_len
        self.num_betas = num_betas
        self.num_pca_comps = num_pca_comps
        self.sdf_len = sdf_len
        aligns = []
        for action in actions:
            align = natsorted(glob.glob(os.path.join(self.anno_folder, '{}/*/anno.pkl'.format(action))))
            # print(action,len(align))
            for a in align:
                aligns.append(a)
        print('total aligns', len(align))
        # Open the file in binary read mode
        with open(config.dataset, 'rb') as file:
            # Load the data from the file using pickle.load() function
            data = pickle.load(file)
        
        self.scene_list = []
        with open(config.scene_list, 'r') as file:
            self.scene_list = file.read().splitlines()
        # print(data.keys())
        # print(data['train'])
        data_new = []
        if phase == 'train':
            aligns_train = data['train']
            align_new = []
            for align in aligns:
                parts = align.split('/')[-2]
                if parts in aligns_train:
                    
                    align_new.append(align)
            
            align_new = align_new
  
            motion_data = self._get_anno_split(align_new)
        elif phase == 'test':
            
            aligns_test = data['seen_scene_unseen_motion'].copy()
                        
            align_new = []
            for align in aligns:
                # print(align)
                parts = align.split('/')[-2]
                if parts in aligns_test:
                    data_new.append(parts)
                    align_new.append(align)
            motion_data = self._get_anno_split(align_new)
        else:
            raise Exception('Unexpected phase.')
        

        print(phase, len(align_new))
        self.hsdf_list = None
        self.msdf_list = None
        with open(config.dataset_hsdf, 'rb') as f:
            self.hsdf_list = pickle.load(f)
        with open(config.dataset_msdf, 'rb') as f:
            self.msdf_list = pickle.load(f)

        
        self.dct_n = config.dct_n
        self.dct_m_in, _ = get_dct_matrix(config.history_len + config.future_len)
        self.dct_m_out, _ = get_dct_matrix(config.history_len + config.future_len)
        self.pad_idx = np.repeat([config.history_len - 1], config.future_len)
        self.i_idx = np.append(np.arange(0,config.history_len),self.pad_idx)


        ## prepare motion data
        self._prepare_motion_data_list(motion_data)

        print("prepare motion data done!!")
        self.motion_data_length = []
        self._split_motion_data(self.motion_data_list, self.motion_length)
        # self._pad_scene_sdf()
        
        self.motion_data_length = np.array(self.motion_data_length)
        print("the average length is", np.mean(self.motion_data_length))
        Console.log('Total {} motion data examples in {} set.'.format(len(self.motion_data_list), self.phase))

    def _get_anno_split(self, align_annos: list):
        self.anno_list = []
        self.scene_data = {}
        
        motion_data = {}

        for a in align_annos:
            with open(a, 'rb') as fp:
                data = pickle.load(fp)
            
            for i, p in enumerate(data):
                action = p['action']
                scene_id = p['scene']
                motion_id = p['motion']
                scene_id = scene_id.split('_')[0]
                if scene_id not in self.scene_list:
                    continue
                mid = a.split('/')[-2]
                mid = mid+'_'+str(i)
                p['mid'] = mid
                self.anno_list.append(p)

                if scene_id not in self.scene_data.keys():
                    scene_data = self._get_scene_sdf(scene_id)
                    # print(scene_data[0])
                    # print(scene_id)
                    self.scene_data[scene_id] = scene_data
                
                if motion_id not in motion_data:
                    with open(os.path.join(self.motion_folder, action, motion_id, 'motion.pkl'), 'rb') as fp:
                        mdata = pickle.load(fp)
                    motion_data[motion_id] = mdata
        
        return motion_data
   
    def _get_scene_sdf(self, scene_id):
        # scene_id should be like:
        zp_path = os.path.join(self.scene_folder, scene_id+'_zp')
        mp_path = os.path.join(self.scene_folder, scene_id+'_mp')
        if os.path.exists(zp_path):
            sdf_path = os.path.join(zp_path, "{}_fsdf.npy".format(scene_id))
            info_path = os.path.join(zp_path, "{}_info.pkl".format(scene_id))
        elif os.path.exists(mp_path):
            sdf_path = os.path.join(mp_path, "{}_fsdf.npy".format(scene_id))
            info_path = os.path.join(mp_path, "{}_info.pkl".format(scene_id))
        else:
            print("there is no sdf file for {}!!!".format(scene_id))
            return
        sdf = np.load(sdf_path)
        sdf = sdf.astype(np.float32)
        with open(info_path,"rb") as f:
            info = pickle.load(f)
        # print(info.keys())
        # print(info['scene_name'])
        data ={'sdf':sdf, "radius": info['radius'], "vsize": info['vsize'], "scene_id": info['scene_name'], "grid":info['grid']}
        return data

    def _pad_scene_sdf(self):
        maxx = 0
        maxy = 0
        maxz = 0
        for scene_id in self.scene_data.keys():
            scene_d = self.scene_data[scene_id]
            vsize = scene_d['vsize']
            maxx = max(vsize[0], maxx)
            maxy = max(vsize[1], maxy)
            maxz = max(vsize[2], maxz)
        desired_shape = (maxx, maxy, maxz)
        print(maxx,maxy,maxz)
        for scene_id in self.scene_data.keys():
            # Determine the amount of padding needed in each dimension
            scene_d = self.scene_data[scene_id]
            vsize = scene_d['vsize']
            crop_sdf = scene_d['sdf']
            padding = []
            for i in range(3):
                diff = desired_shape[i] - crop_sdf.shape[i]
                pad_before = 0
                pad_after = diff 
                padding.append((pad_before, pad_after))

            crop_sdf = np.pad(crop_sdf, padding, mode='constant')
            self.scene_data[scene_id]['pad_sdf'] = crop_sdf

    def _get_anchor_frame_index(self, action: str):
        if action == 'sit':
            return -1
        elif action == 'stand up':
            return 0
        elif action == 'walk':
            return -1
        elif action == 'lie':
            return -1
        else:
            raise Exception('Unexcepted action type.')
    
    def _prepare_motion_data_list(self, motion_data):
        """ Prepare motion data by constructing the motion data according to annotation data

        Args:
            motion_data: a dict, all pure motion data, `{motion_id: motion_data, ...}`
        """
        self.motion_data_list = []

        for anno_index in tqdm(range(len(self.anno_list))):
            anno_data = self.anno_list[anno_index]
            
            motion_id = anno_data['motion']
            motion_trans = anno_data['translation']
            motion_rotat = anno_data['rotation']


            anchor_frame_index = self._get_anchor_frame_index(anno_data['action'])

            gender, origin_trans, origin_orient, betas, pose_body, pose_hand, pose_jaw, pose_eye, joints_traj = motion_data[motion_id]
            ## transform smplx bodies in motion sequence, convert smplx parameter, be careful
            cur_trans, cur_orient, cur_pelvis = self._transform_smplx_from_origin_to_sampled_position(
            motion_trans, motion_rotat, origin_trans, origin_orient, joints_traj[:, 0, :], anchor_frame_index)

            ### representation transfer
            ## 1. only use 10 components of betas
            betas = betas[:self.num_betas]

            mdata = (anno_index, cur_trans, cur_orient, betas.copy(), pose_body, pose_hand, cur_pelvis)
            self.motion_data_list.append(mdata)
    
    def _pad_utterance(self, token_id_seq, pad_val: int=0):
        """ Add padding to token id sequnece of utterance

        Args:
            token_id_seq: a sequence of token id
            pad_val: default is 0
        
        Return:
            Padded token id sequence
        """
        if len(token_id_seq) > self.max_lang_len:
            return token_id_seq[0:self.max_lang_len]
        else:
            return token_id_seq + [pad_val] * (self.max_lang_len - len(token_id_seq))

    def _transform_smplx_from_origin_to_sampled_position(
        self,
        sampled_trans: np.ndarray,
        sampled_rotat: np.ndarray,
        origin_trans: np.ndarray,
        origin_orient: np.ndarray,
        origin_pelvis: np.ndarray,
        anchor_frame: int=0,
    ):
        """ Convert original smplx parameters to transformed smplx parameters

        Args:
            sampled_trans: sampled valid position
            sampled_rotat: sampled valid rotation
            origin_trans: original trans param array
            origin_orient: original orient param array
            origin_pelvis: original pelvis trajectory
            anchor_frame: the anchor frame index for transform motion, this value is very important!!!
        
        Return:
            Transformed trans, Transformed orient, Transformed pelvis
        """
        position = sampled_trans
        rotat = sampled_rotat

        T1 = np.eye(4, dtype=np.float32)
        T1[0:2, -1] = -origin_pelvis[anchor_frame, 0:2]
        T2 = Q(axis=[0, 0, 1], angle=rotat).transformation_matrix.astype(np.float32)
        T3 = np.eye(4, dtype=np.float32)
        T3[0:3, -1] = position
        T = T3 @ T2 @ T1

        trans_t = []
        orient_t = []
        for i in range(len(origin_trans)):
            t_, o_ = SMPLX_Util.convert_smplx_verts_transfomation_matrix_to_body(T, origin_trans[i], origin_orient[i], origin_pelvis[i])
            trans_t.append(t_)
            orient_t.append(o_)
        
        trans_t = np.array(trans_t)
        orient_t = np.array(orient_t)
        pelvis_t = transform_points(origin_pelvis, T)
        return trans_t, orient_t, pelvis_t

    def _pad_motion(self, trans: np.ndarray, orient: np.ndarray, pose_body: np.ndarray, pose_hand: np.ndarray):
        """ Add padding to smplx parameter sequence

        Args:
            trans:
            orient:
            pose_body:
            pose_hand:
        
        Return:
            Padded smplx parameters, i.e. trans, orient, pose_body, pose_hand, and a mask array
        """
        if trans.shape[0] > self.max_motion_len:
            trans = trans[0:self.max_motion_len]
            orient = orient[0:self.max_motion_len]
            pose_body = pose_body[0:self.max_motion_len]
            pose_hand = pose_hand[0:self.max_motion_len]
        
        S, D = trans.shape
        trans_padding = np.zeros((self.max_motion_len - S, D), dtype=np.float32)
        trans = np.concatenate([trans, trans_padding], axis=0)

        _, D = orient.shape
        orient_padding = np.zeros((self.max_motion_len - S, D), dtype=np.float32)
        orient = np.concatenate([orient, orient_padding], axis=0)

        _, D = pose_body.shape
        pose_body_padding = np.zeros((self.max_motion_len - S, D), dtype=np.float32)
        pose_body = np.concatenate([pose_body, pose_body_padding], axis=0)

        _, D = pose_hand.shape
        pose_hand_padding = np.zeros((self.max_motion_len - S, D), dtype=np.float32)
        pose_hand = np.concatenate([pose_hand, pose_hand_padding], axis=0)

        ## generate mask
        motion_mask = np.zeros(self.max_motion_len, dtype=bool)
        motion_mask[S:] = True

        return trans, orient, pose_body, pose_hand, motion_mask




    def _get_crop_sdf(self, location, scene_data, sdf_len):
        radius = scene_data['radius']
        voxel = 0.04
        sdf = scene_data['sdf']
        self.sdf_len = sdf_len
        # print("sdf len", self.sdf_len)
        # print(voxel)
        bx, by, bz = sdf.shape[0], sdf.shape[1], sdf.shape[2]
        location_temp = location.copy()
        location[0] = location[0] - radius[0]
        location[1] = location[1] - radius[2]
        location[2] = location[2] - radius[4]
        location = location // voxel
        # print("location", location)
        leftx = int(max(0,location[0]-self.sdf_len//2))
        lefty = int(max(0,location[1]-self.sdf_len//2))
        leftz = int(max(0,location[2]-self.sdf_len//2))
        rightx = int(min(bx,location[0] + self.sdf_len//2))
        righty = int(min(by,location[1] + self.sdf_len//2))
        rightz = int(min(bz,location[2] + self.sdf_len//2))
        # print("sdf shape", sdf.shape)
        # print(leftx,lefty,leftz, rightx, righty, rightz)
        crop_sdf = sdf[leftx:rightx,lefty:righty,leftz:rightz]


        index = np.zeros(6)
        new_point = np.array([leftx,lefty,leftz]).astype(np.float32)
        new_point = new_point*voxel 
        new_point[0] = new_point[0] + radius[0]
        new_point[1] = new_point[1] + radius[2]
        new_point[2] = new_point[2] + radius[4]
        index[0] = new_point[0]
        index[1] = new_point[1]
        index[2] = new_point[2]
        new_point1 = np.array([leftx+sdf_len-1,lefty+sdf_len-1,leftz+sdf_len-1]).astype(np.float32)
        new_point1 = new_point1*voxel 
        new_point1[0] = new_point1[0] + radius[0]
        new_point1[1] = new_point1[1] + radius[2]
        new_point1[2] = new_point1[2] + radius[4]
        index[3] = new_point1[0]
        index[4] = new_point1[1]
        index[5] = new_point1[2]
        index[:3] -= location_temp
        index[3:] -= location_temp


        # Define the desired shape of the padded volume
        desired_shape = (self.sdf_len, self.sdf_len, self.sdf_len)

        # Determine the amount of padding needed in each dimension
        padding = []
        for i in range(3):
            diff = desired_shape[i] - crop_sdf.shape[i]
            pad_before = 0
            pad_after = diff 
            padding.append((pad_before, pad_after))
     
        crop_sdf = np.pad(crop_sdf, padding, mode='constant')


        return crop_sdf, index
   

    def _split_motion_data(self, motion_data_list, motion_length):
        self.motion_data_list = []
        for data in motion_data_list:
            anno_index, trans, orient, betas, pose_body, pose_hand, pelvis = data
            motion_len = trans.shape[0]
            self.motion_data_length.append(trans.shape[0])
            if motion_len < self.motion_len:
                continue
            for i in range(0,motion_len,10):
                if i + motion_length > motion_len:
                    break


                anno_index_clip = anno_index
                trans_clip = trans[i:i+motion_length,:]
                orient_clip = orient[i:i+motion_length,:]
                betas_clip = betas
                pose_body_clip = pose_body[i:i+motion_length,:]
                pose_hand_clip = pose_hand[i:i+motion_length,:]
                pelvis_clip = pelvis[i:i+motion_length,:]

                mid = self.anno_list[anno_index_clip]['mid']
                mid = mid + '_' + str(i)
                

                hsdf_clip = self.hsdf_list[mid]
                motion_sdf_clip = self.msdf_list[mid]
                
               



                # [T,N]
                motion_hsdf_clip = hsdf_clip
          


                # dct_n, N
                in_hsdf_clip = np.matmul(self.dct_m_in[:self.dct_n, :], motion_hsdf_clip[self.i_idx, :])
                out_hsdf_clip = np.matmul(self.dct_m_in[:self.dct_n, :], motion_hsdf_clip)

                # N dct_n
                in_hsdf_clip = in_hsdf_clip.transpose()
                out_hsdf_clip = out_hsdf_clip.transpose()


                                # dct_n, N

                in_sdf_clip = np.matmul(self.dct_m_in[:self.dct_n, :], motion_sdf_clip[self.i_idx, :])
                out_sdf_clip = np.matmul(self.dct_m_in[:self.dct_n, :], motion_sdf_clip)

                # N dct_n
                in_sdf_clip = in_sdf_clip.transpose()
                out_sdf_clip = out_sdf_clip.transpose()
                

                
                mdata = (anno_index_clip, trans_clip, orient_clip, betas_clip, pose_body_clip, pose_hand_clip, in_sdf_clip,out_sdf_clip, motion_sdf_clip, in_hsdf_clip,out_hsdf_clip, hsdf_clip, mid)
                self.motion_data_list.append(mdata)

   
    def __getitem__(self, index):
        ## get motion data, e.g. smplx parameter sequence
        ## smplx parameters are pre-processed, need to add padding
        anno_index, trans, orient, betas, pose_body, pose_hand,in_sdf,out_sdf, motion_sdf,in_hsdf,out_hsdf, motion_hsdf, mid = self.motion_data_list[index]
        anno_data = self.anno_list[anno_index]

        ## process scene
        scene_id = anno_data['scene']
        scene_id = scene_id.split('_')[0]
        scene_trans = anno_data['scene_translation']
        scene_radius = self.scene_data[scene_id]['radius']
        # scene_wsdf = self._pad_scene_sdf(self.scene_data[scene_id])
        scene_data = self.scene_data[scene_id]
        
        orient_6D = GeometryTransformer.convert_to_6D_rot(torch.tensor(orient)).numpy()
        trans = trans.astype(np.float32)

        trans = trans - scene_trans
        # print("*******************")
        # print(trans)
        motion_center = trans[config.history_len-1]
        # print(motion_center)
        scene_sdf, scene_index = self._get_crop_sdf(motion_center.copy(), scene_data, sdf_len = 100)
        trans = trans - motion_center
        return mid, scene_radius, scene_trans, motion_center, scene_sdf,  trans, orient_6D, betas, pose_body, pose_hand, motion_sdf, scene_index, in_sdf, out_sdf, motion_hsdf, in_hsdf

    def __len__(self):
        return len(self.motion_data_list)
    


def collate_random_motion(data):
    (scene_id, scene_radius,  scene_trans, motion_transformation, scene_sdf,  trans, orient, betas, pose_body, pose_hand,motion_sdf, scene_index, in_sdf, out_sdf, motion_hsdf, in_hsdf) = zip(*data)

    # scene_wsdf = np.asarray(scene_wsdf).astype(np.float32)
    ## convert to tensor

    # print(type(scene_wsdf))
    trans = torch.FloatTensor(np.array(trans))
    orient = torch.FloatTensor(np.array(orient))
    betas = torch.FloatTensor(np.array(betas))
    pose_body = torch.FloatTensor(np.array(pose_body))
    pose_hand = torch.FloatTensor(np.array(pose_hand))
    motion_transformation = torch.FloatTensor(np.array(motion_transformation))
    scene_sdf = torch.FloatTensor(np.array(scene_sdf))
    scene_trans = torch.FloatTensor(np.array(scene_trans))
    scene_radius = torch.FloatTensor(np.array(scene_radius))
    motion_sdf = torch.FloatTensor(np.array(motion_sdf))
    scene_index = torch.FloatTensor(np.array(scene_index))
    in_sdf = torch.FloatTensor(np.array(in_sdf))
    out_sdf = torch.FloatTensor(np.array(out_sdf))
    motion_hsdf = torch.FloatTensor(np.array(motion_hsdf))
    in_hsdf = torch.FloatTensor(np.array(in_hsdf))
    # scene_wsdf = torch.FloatTensor(np.array(scene_wsdf))

    batch = (
        scene_id,
        scene_radius.cuda(),
        scene_trans.cuda(),
        motion_transformation.cuda(),
        scene_sdf.cuda(),
        trans.cuda(),
        orient.cuda(),
        betas.cuda(),
        pose_body.cuda(),
        pose_hand.cuda(),
        motion_sdf.cuda(),
        scene_index.cuda(),
        in_sdf.cuda(),
        out_sdf.cuda(),
        motion_hsdf.cuda(),
        in_hsdf.cuda()
    )

    return batch