import argparse
import os
import time

###############NEED UPDATE######################################################

smplx_folder = '/home/chaoyue/LHMS/models_smplx_v1_1/models/'

pure_motion_folder = '/hdd/DATA_MULTUL_DISTANCE/HUMANISE/HUMANISE_MOTION/pure_motion/'
align_data_folder = '/hdd/DATA_MULTUL_DISTANCE/HUMANISE/HUMANISE_MOTION/align_data_release/'

sdf_folder = '/hdd/Pointnet2.ScanNet/possion_reconstruction/scenedone4'
dataset = '/hdd/ALL_final_model/MutualDistance/HUMANISE/dataset.pkl'
dataset_scene_points = '/hdd/ALL_final_model/MutualDistance/HUMANISE/scene_points_150_2.pkl'
scene_list = '/hdd/ALL_final_model/MutualDistance/HUMANISE/scenelist_keep.txt'
dataset_hsdf = '/hdd/ALL_final_model/MutualDistance/HUMANISE/dataset_hsdf.pkl'
dataset_msdf = '/hdd/ALL_final_model/MutualDistance/HUMANISE/dataset_msdf.pkl'


##################################################################################

## train & test
batch_size = 2
learning_rate = 5e-5
num_epoch = 7
device = 'cuda'
num_workers = 0
weight_loss_rec = 1.0
weight_loss_rec_pose = 1.0
weight_loss_rec_vertex = 1.0
weight_loss_kl = 0.1
weight_loss_vposer = 1e-3
weight_loss_ground = 1.0
action = 'sit'

## model setting
pretrained_scene_model = ''
lang_feat_size = 768
scene_feat_size = 512
scene_group_size = 16 # pointnet++ final output size
max_lang_len = 32
motion_len = 45
model_hidden_size = 512
model_condition_size = 512
model_z_latent_size = 32
npoints = 8192

input_feature_gcn = 311
hidden_feature_gcn = 256
p_dropout = 0.5
dct_n = 45
hsdf_in =150

nscene_point = 150
scene_in = 1
scene_out = 256
sdf_shape = 67
sdf_in = 67
sdf_out = 128
motion_shape = 162 # to be done
motion_in = 256
motion_out = 256
sdf_len = 100
history_len = 15
future_len = 30
weight_loss_rec = 1.0
weight_loss_rec_body_pose = 0.5
weight_loss_rec_hand_pose = 0.1
weight_loss_sdf = 1.0
weight_loss_hsdf = 1.0
weight_loss_pred_hsdf =0.5
weight_loss_pred_msdf = 0.5
resume_model ='/hdd/ALL_final_model/HUMANISE/GCN+Motion_Scene_GRUSDF_beta_grids_step_lr5e4_hsdf/kaolin/logs_temp/20231112_083514'
resume_model_s1 = '/home/chaoyue/LHMS/GCN+Motion_Scene_GRUSDF_beta_grids_step_lr5e4_hsdf/kaolin/logs_result/20231111_153403'
resume_model_s2 = '/home/chaoyue/LHMS/Motion_Scene_GRUSDF_beta_grids_step_lr5e4_hsdf_150/logs_temp/20231111_141240'

## smplx
num_pca_comps = 12
num_betas = 10
gender = 'neutral'
bigdata = False

def parse_args():
    parser = argparse.ArgumentParser()

    ## path setting
    parser.add_argument('--log_dir', 
                        type=str, 
                        default=os.path.join(os.getcwd(), 'logs_temp/'),
                        help='dir for saving checkpoints and logs')
    parser.add_argument('--stamp', 
                        type=str, 
                        default=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                        help='timestamp')
    ## train & test setting
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=batch_size,
                        help='batch size to train')
    parser.add_argument('--lr', 
                        type=float, 
                        default=learning_rate,
                        help='initial learing rate')
    parser.add_argument('--num_epoch', 
                        type=int, 
                        default=num_epoch,
                        help='#epochs to train')
    parser.add_argument('--device', 
                        type=str, 
                        default=device,
                        help='set device for training')
    parser.add_argument('--resume_model',
                        type=str,
                        default=resume_model,
                        help='resume model path')
    parser.add_argument('--resume_model_s1',
                        type=str,
                        default=resume_model_s1,
                        help='resume model path')
    parser.add_argument('--resume_model_s2',
                        type=str,
                        default=resume_model_s2,
                        help='resume model path')
    parser.add_argument('--num_workers',
                        type=int,
                        default=num_workers,
                        help='number of dataloader worker processer')
    parser.add_argument('--all_body_vertices',
                        action="store_true",
                        help='use all body vertices to regress')
    parser.add_argument('--action',
                        type=str,
                        default=action,
                        help='action type')
    ## model setting
    parser.add_argument('--pretrained_scene_model',
                        type=str,
                        default=pretrained_scene_model,
                        help='pre-trained scene model')
    parser.add_argument('--lang_feat_size',
                        type=int,
                        default=lang_feat_size,
                        help='language feature size')
    parser.add_argument('--scene_feat_size',
                        type=int,
                        default=scene_feat_size,
                        help='scene feature size')
    parser.add_argument('--scene_group_size',
                        type=int,
                        default=scene_group_size,
                        help='scene group size')
    parser.add_argument('--use_color',
                        action="store_true",
                        help='use point rgb color')
    parser.add_argument('--use_normal',
                        action="store_true",
                        help='use point normal')
    parser.add_argument('--max_lang_len',
                        type=int,
                        default=max_lang_len,
                        help='max length of language description')
    parser.add_argument('--motion_len',
                        type=int,
                        default=motion_len,
                        help='max length of motion sequence')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=model_hidden_size,
                        help='the size the hidden state in CVAE model')
    parser.add_argument('--condition_latent_size',
                        type=int,
                        default=model_condition_size,
                        help='the size the condition latent')
    parser.add_argument('--z_size',
                        type=int,
                        default=model_z_latent_size,
                        help='the size the z latent')
    parser.add_argument('--npoints',
                        type=int,
                        default=npoints,
                        help='sample points number of pointcloud')
    parser.add_argument('--nscene_point',
                        type=int,
                        default=nscene_point,
                        help='number of points in scene to describe the human motion')
    parser.add_argument('--scene_in',
                        type=int,
                        default=scene_in,
                        help='input size of the scenenet')
    parser.add_argument('--scene_out',
                        type=int,
                        default=scene_out,
                        help='output size of the scenenet')
    parser.add_argument('--sdf_shape',
                        type=int,
                        default=sdf_shape,
                        help='output size of the scenenet')
    parser.add_argument('--p_dropout',
                        type=float,
                        default=p_dropout,
                        help='output size of the scenenet')
                        
    parser.add_argument('--hidden_feature_gcn',
                        type=int,
                        default=hidden_feature_gcn,
                        help='output size of the scenenet')
                        
    parser.add_argument('--sdf_in',
                        type=int,
                        default=sdf_in,
                        help='output size of the scenenet')
    parser.add_argument('--sdf_out',
                        type=int,
                        default=sdf_out,
                        help='output size of the scenenet')
    parser.add_argument('--motion_shape',
                        type=int,
                        default=motion_shape,
                        help='output size of the scenenet')
    parser.add_argument('--motion_in',
                        type=int,
                        default=motion_in,
                        help='output size of the scenenet')
    parser.add_argument('--motion_out',
                        type=int,
                        default=motion_out,
                        help='output size of the scenenet')
    parser.add_argument('--history_len',
                        type=int,
                        default=history_len,
                        help='output size of the scenenet')
    parser.add_argument('--future_len',
                        type=int,
                        default=future_len,
                        help='output size of the scenenet')
    parser.add_argument('--dct_n',
                        type=int,
                        default=dct_n,
                        help='output size of the scenenet')
    parser.add_argument('--hsdf_in', 
                        type=int, 
                        default=hsdf_in,
                        help='#epochs to train')
    parser.add_argument('--weight_loss_rec',
                        type=float,
                        default=weight_loss_rec,
                        help='output size of the scenenet')
    parser.add_argument('--weight_loss_rec_body_pose',
                        type=float,
                        default=weight_loss_rec_body_pose,
                        help='output size of the scenenet')
    parser.add_argument('--weight_loss_rec_hand_pose',
                        type=float,
                        default=weight_loss_rec_hand_pose,
                        help='output size of the scenenet')
    parser.add_argument('--weight_loss_sdf',
                        type=float,
                        default=weight_loss_sdf,
                        help='output size of the scenenet')
    parser.add_argument('--weight_loss_hsdf',
                        type=float,
                        default=weight_loss_hsdf,
                        help='output size of the scenenet')
                        
    parser.add_argument('--weight_loss_pred_hsdf',
                        type=float,
                        default=weight_loss_pred_hsdf,
                        help='output size of the scenenet')
    parser.add_argument('--weight_loss_pred_msdf',
                        type=float,
                        default=weight_loss_pred_hsdf,
                        help='output size of the scenenet')
    ## smplx setting
    parser.add_argument('--num_pca_comps', 
                        type=int, 
                        default=num_pca_comps,
                        help='number of pca component of hand pose')
    parser.add_argument('--num_betas', 
                        type=int, 
                        default=num_betas,
                        help='number of pca component of body shape beta')
    parser.add_argument("--sdf_len",
                        type = int,
                        default=sdf_len,
                        help="the length of cropped sdf")
    parser.add_argument('--vis',
                        action="store_true",
                        help='visualize the prediction')
    
    parser.add_argument('--sdata',
                        action="store_true",
                        help='small dataset')
    
    args = parser.parse_args()

    return args
