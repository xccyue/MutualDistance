import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
class DatasetGTA(Dataset):

    def __init__(self, mode, dataset_specs):
        self.mode = mode
        self.t_his = t_his = dataset_specs.get('t_his',30)
        self.t_pred = t_pred = dataset_specs.get('t_pred',60)
        self.t_total = t_his + t_pred
        self.random_rot = dataset_specs.get('random_rot',False)
        self.is_contact = dataset_specs.get('is_contact',False)
        self.is_frame_contact = dataset_specs.get('is_frame_contact',False)
        self.step = step = dataset_specs.get('step',5)
        self.num_scene_points = num_scene_points = dataset_specs.get('num_scene_points', 10000)
        self.max_dist_from_human = max_dist_from_human = dataset_specs.get('max_dist_from_human', 2.5)
        self.wscene = wscene = dataset_specs.get('wscene',True)
        self.wcont = wcont = dataset_specs.get('wcont',True)
        self.num_cont_points = num_cont_points = dataset_specs.get('num_cont_points', 500)
        self.sigma = dataset_specs.get('sigma', 0.02)
        self.cont_thre = 0.2

        self.data_file = data_file = dataset_specs.get('data_file', '/hdd/DATA_MULTUL_DISTANCE/GTA_IM_MOTION_SCENE/GTA-IM/data_v2_downsample0.02')
        # self.scene_split = {'train': ['r001','r002','r003', 'r006'],

        #                     'test': ['r010', 'r011', 'r013']
        #                     }
        self.scene_split = {'train': ['r001','r002','r003', 'r006'],

                            'test': ['r010', 'r011', 'r013']
                            }

        self.pose = {}
        self.scene = {}
        self.idx2scene = {}

        print('read original data file')
        for i, seq in tqdm(enumerate(os.listdir(data_file))):
            # if '2020-06-04-22-57-20_r013_sf0' not in seq:
            #     continue
            room = seq.split('_')[1]
            if room not in self.scene_split[mode]:
                continue
            data_tmp=np.load(f'{data_file}/{seq}',allow_pickle=True)
            self.pose[i] = data_tmp['joints']
            self.idx2scene[i] = seq[:-4]
            # if wscene or wcont:
            self.scene[i] = data_tmp['scene_points']

            ########### for debug
            # if len(self.pose) > 0:
            #     break
        

        self.data = {}
        self.scene_point_idx = {}
        self.cont_idx = {}
        self.sdf_coord_idxs = {}
        k = 0
        min_num_scene = 1000000
        max_num_scene = 0
        print("generateing data idxs")
        for sub in tqdm(self.pose.keys()):
            room = self.idx2scene[sub].split('_')[1]
            seq_len = self.pose[sub].shape[0]
            idxs_frame = np.arange(0,seq_len - self.t_total + 1,step)
            for i in idxs_frame:
                self.data[k] = f'{sub}.{i}'
                k = k + 1
        

        for idx in range(len(self.data.keys())):
            item_key = self.data[idx].split('.')
            sub = int(item_key[0])
            fidx = int(item_key[1])
            subj = self.idx2scene[sub]
            room = subj.split('_')[1]
            item_key = f"{subj}.{item_key[1]}"
            # if item_key not in self.dis_data.keys():
            #     print(self.mode, 'there is no!!!!!!!!', item_key)

        # print(f"num of scene points from {min_num_scene:d} to {max_num_scene:d}")
        print(f'seq length {self.t_total},in total {k} seqs')


        print("read sdf")

        self.sdf_file = '/hdd/DATA_MULTUL_DISTANCE/GTA_IM_MOTION_SCENE/GTA-IM/scene_sdf'
        self.sdf = {}
        # print(os.listdir(self.sdf_file))
        for i, seq in tqdm(enumerate(os.listdir(self.sdf_file))):
            # if '2020-06-04-22-57-20_r013_sf0' not in seq:
            #     continue
            room = seq.split('_')[0]
            # print('room', room)
            if room not in self.scene_split[mode]:
                continue
            if not seq.endswith('.npz'):
                continue
            print('room1', room)
            data_tmp=np.load(f'{self.sdf_file}/{seq}',allow_pickle=True)
            # print(data_tmp.keys())
            da = {}
            da['xs'] = data_tmp['xs']
            da['ys'] = data_tmp['ys']
            da['zs'] = data_tmp['zs']
            da['sdf'] = data_tmp['sd'] * -1.0
            da['xs_max'] = np.max(da['xs'])
            da['xs_min'] = np.min(da['xs'])
            da['ys_max'] = np.max(da['ys'])
            da['ys_min'] = np.min(da['ys'])
            da['zs_max'] = np.max(da['zs'])
            da['zs_min'] = np.min(da['zs'])

            print((np.max(da['xs']) - np.min(da['xs']))/da['sdf'].shape[0])
            self.sdf[room] = da
            # print(np.max(self.sx[i]), np.min(self.sx[i]))
            # print(np.max(self.sy[i]), np.min(self.sy[i]))
            # print(self.sdf[i].shape)

            # self.pose[i] = data_tmp['joints']
            # self.idx2scene[i] = seq[:-4]
            # # if wscene or wcont:
            # self.scene[i] = data_tmp['scene_points']

            # ########### for debug
            # if len(self.pose) > 0:
            #     break


    def __len__(self):
        return len(list(self.data.keys()))


    def __getitem__(self, idx):

        item_key = self.data[idx].split('.')
        sub = int(item_key[0])
        fidx = int(item_key[1])
        subj = self.idx2scene[sub]
        room = subj.split('_')[1]
        item_key = f"{subj}.{item_key[1]}"
        scene_data = self.sdf[room]

        pose = torch.tensor(self.pose[sub][fidx:fidx + self.t_total]).float()
        pose_rotate = pose.clone()
        scene_origin = torch.clone(pose[self.t_his-1,14:15])
        if self.mode == 'train':
            degree = torch.rand(1).item() * 360
            # degree = 0
        else:
            degree = 0
        
        R = self._rotation_matrix([0,0,1],degree)
        R = torch.tensor(R).float()
        R_ = R.clone()

        S,N,_ = pose_rotate.shape
        pose_rotate = torch.mm(R,pose_rotate.view(-1,3).transpose(0,1)).transpose(0,1).view(S,N,3)
        scene_origin_rotate = torch.clone(pose_rotate[self.t_his-1,14:15])

        # scene_sdf_rotate, index_rotate = self._rotate_crop_sdf(scene_origin.detach().cpu().numpy().copy(),scene_origin_rotate.detach().cpu().numpy().copy(),scene_data,100,R_)

        pose_rotate = pose_rotate - scene_origin_rotate # [90, 21, 3]
        scene_sdf_origin, index1, index2 = self._get_origin_sdf(scene_origin.detach().cpu().numpy().copy(),scene_data,100)


        return pose_rotate, scene_origin_rotate, item_key, scene_sdf_origin, index1, index2, R_
        # return pose, scene_sdf, scene_origin, item_key, index
    

    def _rotate_crop_sdf(self, location_origin,location, scene_data, sdf_len, R):
        # radius = scene_data['radius']
        # voxel = 0.05
        scene_sdf = scene_data['sdf']
        
        self.sdf_len = sdf_len
        location = location.reshape(3)


        # Define the volume's parameters
        resolution = 0.05
        length = 5.0
        half_length = length / 2
        num_points = int(length / resolution)

        # Generate a range of offsets with the given resolution
        offsets = np.linspace(-half_length, half_length, num_points, endpoint=False)

        # Create a 3D meshgrid for the x, y, z coordinates
        X, Y, Z = np.meshgrid(offsets, offsets, offsets, indexing='ij')

        # Adjust by the center point values
        X += location[0]
        Y += location[1]
        Z += location[2]

        # Stack the coordinates to get the [100, 100, 100, 3] shape
        volume = np.stack((X, Y, Z), axis=-1)

        volume = torch.tensor(volume).float()
        index = torch.zeros(6)
        index[:3] = volume[0,0,0]
        index[3:] = volume[99,99,99]
        index[:3] -= location
        index[3:] -= location

        volume = volume.view(-1,3).transpose(0,1)
        R_inverse = torch.linalg.inv(R)
        volume = R_inverse@volume
        volume = volume.transpose(0,1)
        volume = volume.reshape(1,100*100*100,3)

        scene_sdf1, index1, index2 = self._get_origin_sdf(location_origin, scene_data, sdf_len)

        volume = 2*((volume - index1)/(index2 - index1) - 0.5)
        # print(volume)
        scene_sdf1 = scene_sdf1.unsqueeze(0).float()
        volume = volume.float()


        # Find the points (rows) where any coordinate is out of the [-1, 1] range
        out_of_range_mask = (volume > 1) | (volume < -1)
        out_of_range_rows = torch.any(out_of_range_mask, dim=2)
        out_of_range_indices = torch.where(out_of_range_rows)[1]  # We take the second dimension since the first is always 0 due to the shape [1, 100, 3]

        # Find the specific coordinates (XYZ) which are out of range for each point
        out_of_range_coords = torch.where(out_of_range_mask)
        point_indices, coord_indices = out_of_range_coords[1], out_of_range_coords[2]
        # Set the values of the out-of-range points to 0
        volume[0, out_of_range_indices] = 0
        # print(torch.max(volume),torch.min(volume))

   
        sdf_volume = F.grid_sample(scene_sdf1[:,None],volume[:,None,None,:,[2,1,0]], align_corners=True)
       
        
        sdf_volume = sdf_volume.reshape(1,100*100*100)

        sdf_volume[0,out_of_range_indices] = 0
        # print(sdf_volume.shape)
        

        sdf_volume = sdf_volume.squeeze().reshape(100,100,100)


        return sdf_volume, index
    
    def _rotation_matrix(self,axis, theta_deg):
        """
        Returns the rotation matrix for a rotation about a given axis by a specified angle in degrees.
        
        Parameters:
        - axis: A list or array specifying the axis of rotation [x, y, z].
        - theta_deg: The rotation angle in degrees.
        
        Returns:
        - 3x3 numpy array representing the rotation matrix.
        """
        
        theta = np.radians(theta_deg)  # Convert angle from degrees to radians
        axis = np.asarray(axis)
        axis = axis/np.linalg.norm(axis)  # Normalize the axis
        
        c = np.cos(theta)
        s = np.sin(theta)
        C = 1 - c
        x, y, z = axis
        
        # Elements of the rotation matrix
        R = np.array([
            [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
        ])
        
        return R
    def _get_origin_sdf(self, location, scene_data, sdf_len):
        sdf = scene_data['sdf']
        sdf_len = 2*sdf_len
        location = location.reshape(3)
        bx, by, bz = sdf.shape[0], sdf.shape[1], sdf.shape[2]
        
        # print('location', location)
        location[0] = int(((location[0] - scene_data['xs_min']) /(scene_data['xs_max'] - scene_data['xs_min'])) * sdf.shape[0])
        location[1] = int(((location[1] - scene_data['ys_min']) /(scene_data['ys_max'] - scene_data['ys_min'])) * sdf.shape[1])
        location[2] = int(((location[2] - scene_data['zs_min']) /(scene_data['zs_max'] - scene_data['zs_min'])) * sdf.shape[2])
       

        
        # print("location", location)
        leftx = int(max(0,location[0]-sdf_len//2))
        lefty = int(max(0,location[1]-sdf_len//2))
        leftz = int(max(0,location[2]-sdf_len//2))
        rightx = int(min(bx,location[0] + sdf_len//2))
        righty = int(min(by,location[1] + sdf_len//2))
        rightz = int(min(bz,location[2] + sdf_len//2))
        # print("sdf shape", sdf.shape)
        # print(leftx,lefty,leftz, rightx, righty, rightz)
        crop_sdf = sdf[leftx:rightx,lefty:righty,leftz:rightz]

        # Define the desired shape of the padded volume
        desired_shape = (sdf_len, sdf_len, sdf_len)

        # Determine the amount of padding needed in each dimension
        padding = []
        for i in range(3):
            diff = desired_shape[i] - crop_sdf.shape[i]
            if i == 0:
                pad_before = sdf_len//2 - int(location[0] - leftx)
                pad_after = sdf_len//2 - int(rightx - location[0])
            elif i == 1:
                pad_before = sdf_len//2 - int(location[1] - lefty)
                pad_after = sdf_len//2 - int(righty - location[1])
            else:
                pad_before = sdf_len//2 - int(location[2] - leftz)
                pad_after = sdf_len//2 - int(rightz - location[2])
            
            padding.append((pad_before, pad_after))
            if pad_after + pad_before != diff:
                print('error')

     
        crop_sdf = np.pad(crop_sdf, padding, mode='constant')


        index1 = np.zeros(3)
        new_point = np.array([location[0]-sdf_len//2,location[1]-sdf_len//2,location[2]-sdf_len//2]).astype(np.float32)
        new_point[0] = (new_point[0] / sdf.shape[0])*(scene_data['xs_max'] - scene_data['xs_min']) +  scene_data['xs_min']
        new_point[1] = (new_point[1] / sdf.shape[1])*(scene_data['ys_max'] - scene_data['ys_min']) +  scene_data['ys_min']
        new_point[2] = (new_point[2] / sdf.shape[2])*(scene_data['zs_max'] - scene_data['zs_min']) +  scene_data['zs_min']

        index1[0] = new_point[0]
        index1[1] = new_point[1]
        index1[2] = new_point[2]
        new_point1 = np.array([location[0]+sdf_len//2,location[1]+sdf_len//2,location[2]+sdf_len//2]).astype(np.float32)

        new_point = new_point1
        new_point[0] = (new_point[0] / sdf.shape[0])*(scene_data['xs_max'] - scene_data['xs_min']) +  scene_data['xs_min']
        new_point[1] = (new_point[1] / sdf.shape[1])*(scene_data['ys_max'] - scene_data['ys_min']) +  scene_data['ys_min']
        new_point[2] = (new_point[2] / sdf.shape[2])*(scene_data['zs_max'] - scene_data['zs_min']) +  scene_data['zs_min']
        index2 = np.zeros(3)
        index2[0] = new_point[0]
        index2[1] = new_point[1]
        index2[2] = new_point[2]

        crop_sdf = torch.tensor(crop_sdf)
        index1 = torch.tensor(index1)
        index2 = torch.tensor(index2)

        return crop_sdf, index1, index2
    def _get_crop_sdf(self, location, scene_data, sdf_len):
        # radius = scene_data['radius']
        # voxel = 0.05
        sdf = scene_data['sdf']
        self.sdf_len = sdf_len
        location = location.reshape(3)
        # print("sdf len", self.sdf_len)
        # print(voxel)
        bx, by, bz = sdf.shape[0], sdf.shape[1], sdf.shape[2]
        location_temp = location.copy()
        # print('location', location)
        location[0] = int(((location[0] - scene_data['xs_min']) /(scene_data['xs_max'] - scene_data['xs_min'])) * sdf.shape[0])
        location[1] = int(((location[1] - scene_data['ys_min']) /(scene_data['ys_max'] - scene_data['ys_min'])) * sdf.shape[1])
        location[2] = int(((location[2] - scene_data['zs_min']) /(scene_data['zs_max'] - scene_data['zs_min'])) * sdf.shape[2])
       

        
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
        new_point[0] = (new_point[0] / sdf.shape[0])*(scene_data['xs_max'] - scene_data['xs_min']) +  scene_data['xs_min']
        new_point[1] = (new_point[1] / sdf.shape[1])*(scene_data['ys_max'] - scene_data['ys_min']) +  scene_data['ys_min']
        new_point[2] = (new_point[2] / sdf.shape[2])*(scene_data['zs_max'] - scene_data['zs_min']) +  scene_data['zs_min']

        index[0] = new_point[0]
        index[1] = new_point[1]
        index[2] = new_point[2]
        new_point1 = np.array([leftx+sdf_len-1,lefty+sdf_len-1,leftz+sdf_len-1]).astype(np.float32)
        new_point = new_point1
        new_point[0] = (new_point[0] / sdf.shape[0])*(scene_data['xs_max'] - scene_data['xs_min']) +  scene_data['xs_min']
        new_point[1] = (new_point[1] / sdf.shape[1])*(scene_data['ys_max'] - scene_data['ys_min']) +  scene_data['ys_min']
        new_point[2] = (new_point[2] / sdf.shape[2])*(scene_data['zs_max'] - scene_data['zs_min']) +  scene_data['zs_min']
        index[3] = new_point[0]
        index[4] = new_point[1]
        index[5] = new_point[2]
        index[:3] -= location_temp
        index[3:] -= location_temp




        # Define the desired shape of the padded volume
        desired_shape = (self.sdf_len, self.sdf_len, self.sdf_len)

        # Determine the amount of padding needed in each dimension
        padding = []
        for i in range(3):
            diff = desired_shape[i] - crop_sdf.shape[i]
            if i == 0:
                pad_before = self.sdf_len//2 - int(location[0] - leftx)
                pad_after = self.sdf_len//2 - int(rightx - location[0])
            elif i == 1:
                pad_before = self.sdf_len//2 - int(location[1] - lefty)
                pad_after = self.sdf_len//2 - int(righty - location[1])
            else:
                pad_before = self.sdf_len//2 - int(location[2] - leftz)
                pad_after = self.sdf_len//2 - int(rightz - location[2])
            
            padding.append((pad_before, pad_after))
            if pad_after + pad_before != diff:
                print('error')

     
        crop_sdf = np.pad(crop_sdf, padding, mode='constant')


        index = np.zeros(6)
        new_point = np.array([location[0]-self.sdf_len//2,location[1]-self.sdf_len//2,location[2]-self.sdf_len//2]).astype(np.float32)
        new_point[0] = (new_point[0] / sdf.shape[0])*(scene_data['xs_max'] - scene_data['xs_min']) +  scene_data['xs_min']
        new_point[1] = (new_point[1] / sdf.shape[1])*(scene_data['ys_max'] - scene_data['ys_min']) +  scene_data['ys_min']
        new_point[2] = (new_point[2] / sdf.shape[2])*(scene_data['zs_max'] - scene_data['zs_min']) +  scene_data['zs_min']

        index[0] = new_point[0]
        index[1] = new_point[1]
        index[2] = new_point[2]
        new_point1 = np.array([location[0]+self.sdf_len//2-1,location[1]+self.sdf_len//2-1,location[2]+self.sdf_len//2-1]).astype(np.float32)

        new_point = new_point1
        new_point[0] = (new_point[0] / sdf.shape[0])*(scene_data['xs_max'] - scene_data['xs_min']) +  scene_data['xs_min']
        new_point[1] = (new_point[1] / sdf.shape[1])*(scene_data['ys_max'] - scene_data['ys_min']) +  scene_data['ys_min']
        new_point[2] = (new_point[2] / sdf.shape[2])*(scene_data['zs_max'] - scene_data['zs_min']) +  scene_data['zs_min']
        index[3] = new_point[0]
        index[4] = new_point[1]
        index[5] = new_point[2]
        index[:3] -= location_temp
        index[3:] -= location_temp



        return crop_sdf, index



   

def get_sdf_grid_batch( verts, scene_sdf1, scene_index1):
    
    B = verts.shape[0]
    S = verts.shape[1]
    B,S,N,_ = verts.shape
    lo1 = verts.reshape(B,S*N,3)
    scene_index1 = scene_index1.unsqueeze(1)
    lo1 =2*((lo1 - scene_index1[:,:,:3])/(scene_index1[:,:,3:] - scene_index1[:,:,:3]) - 0.5)
    sdf1 = F.grid_sample(scene_sdf1[:,None],lo1[:,None,None,:,[2,1,0]], align_corners=True)
    sdf1 = sdf1.reshape(B,S,N)

    return sdf1


def _get_hsdf(verts, scene_points):
    B = verts.shape[0]
    S = verts.shape[1]
    N = verts.shape[2]
    # print(scene_points.shape)
    verts = verts.reshape(B*S,N,-1)
    # print(verts.shape, scene_point.shape)
    distance = torch.cdist(scene_points,verts)
    distance = torch.min(distance,dim=2)[0]
    distance = distance.reshape(B,S,-1)

    return distance




def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

if __name__ == '__main__':
    dataset = DatasetGTA('test',{})

    generator = DataLoader(dataset, batch_size=2, shuffle=False,
                           num_workers=2, pin_memory=True, drop_last=True)
    result = {}
    with open('../scene_point.pkl', 'rb') as f:
        data = pickle.load(f)

    scene_points = data
    scene_points = scene_points /2 * 2.5
    scene_points = torch.tensor(scene_points).cuda()
    scene_points = scene_points.unsqueeze(0).repeat(2*90,1,1)

    dct_n = 60
    dct_m_in, _ = get_dct_matrix(30 + 60)
    dct_m_out, _ = get_dct_matrix(30 + 60)
    pad_idx = np.repeat([30 - 1], 60)
    i_idx = np.append(np.arange(0,30),pad_idx)
    # print(torch.sqrt(torch.sum(scene_points**2, dim = -1)))
    
    for pose, scene_sdf, scene_origin, item_key, index, scene_vert in (tqdm(generator)):

        B, S, N, _ = pose.shape
        scene_sdf = scene_sdf.cuda()
        pose = pose.cuda()
        index = index.cuda()
        print(pose)
        
        # print('pose input', pose.shape)
        msdf_input = get_sdf_grid_batch(pose,scene_sdf, index)
        hsdf_input = _get_hsdf(pose,scene_points)

        for b in range(B):
            da = {}
            da['pose'] = pose.detach().cpu().numpy()
            da['motion_center'] = scene_origin.detach().cpu().numpy()
            da['index'] = index.detach().cpu().numpy()

            da['msdf_seq'] = msdf_input[b].detach().cpu().numpy()
            da['hsdf_seq'] = hsdf_input[b].detach().cpu().numpy()

            scene = scene_vert[b].detach().cpu().numpy()
            p = pose[b].detach().cpu().numpy()

            
            # Create point cloud objects
            point_cloud1 = o3d.geometry.PointCloud()
            point_cloud1.points = o3d.utility.Vector3dVector(scene.reshape(-1,3))

            point_cloud2 = o3d.geometry.PointCloud()
            point_cloud2.points = o3d.utility.Vector3dVector(p.reshape(S*N,-1))

            # Merge point clouds
            merged_point_cloud = point_cloud1 + point_cloud2
            o3d.io.write_point_cloud("./vis/{}.ply".format(item_key[b]), merged_point_cloud)
            for j in range(N):
                print(da['msdf_seq'].shape)
                print(j, type(j))
                pred_n = msdf_input[b, :, j].detach().cpu().numpy()
                m_name = str(j)
                path_name = "./vis"
                if not os.path.exists(path_name):
                    os.mkdir(path_name)
                plt.plot(pred_n,label = "target")
                plt.xlabel('Frame')
                plt.ylabel("SDF")
                plt.legend()
                
                plt.savefig(os.path.join(path_name,"{}_{}_SDF.png".format(item_key[b], m_name)))
                plt.close()
            
            for j in range(10):
                print(da['msdf_seq'].shape)
                print(j, type(j))
                pred_n = hsdf_input[b, :, j].detach().cpu().numpy()
                m_name = str(j)
                path_name = "./vis"
                if not os.path.exists(path_name):
                    os.mkdir(path_name)
                plt.plot(pred_n,label = "target")
                plt.xlabel('Frame')
                plt.ylabel("HSDF")
                plt.legend()
                
                plt.savefig(os.path.join(path_name,"{}_{}_HSDF.png".format(item_key[b], m_name)))
                plt.close()




            motion_hsdf_clip = hsdf_input[b].detach().cpu().numpy()
            # print('hsdf_clip', motion_hsdf_clip.shape)

            # dct_n, N
            in_hsdf_clip = np.matmul(dct_m_in[:dct_n, :], motion_hsdf_clip[i_idx, :])
            out_hsdf_clip = np.matmul(dct_m_in[:dct_n, :], motion_hsdf_clip)

            # N dct_n
            in_hsdf_clip = in_hsdf_clip.transpose()
            out_hsdf_clip = out_hsdf_clip.transpose()

            da['hsdf_in'] = in_hsdf_clip
            # dct_n, N
            motion_sdf_clip = msdf_input[b].detach().cpu().numpy()

            in_sdf_clip = np.matmul(dct_m_in[:dct_n, :], motion_sdf_clip[i_idx, :])
            out_sdf_clip = np.matmul(dct_m_in[:dct_n, :], motion_sdf_clip)

            # N dct_n
            in_sdf_clip = in_sdf_clip.transpose()
            out_sdf_clip = out_sdf_clip.transpose()


            da['msdf_in'] = in_sdf_clip

            result[item_key[b]] = da





    with open('gta_test_dataset.pkl', 'wb') as f:
        pickle.dump(result, f)

