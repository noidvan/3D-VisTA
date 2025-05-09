from dataset.path_config import SCAN_FAMILY_BASE, MASK_BASE
import json
import os
import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm

from utils.eval_helper import convert_pc_to_box

# rotate angles
ROTATE_ANGLES = [0, np.pi/2, np.pi, np.pi*3/2]

class LoadScannetMixin(object):
    def __init__(self):
        pass
    
    def load_scannet(self, scan_ids, pc_type, load_inst_info):
        scans = {}
        missing_obj_count = 0
        # attribute
        # inst_labels, inst_locs, inst_colors, pcds, / pcds_pred, inst_labels_pred
        for scan_id in tqdm(scan_ids):
            # load inst
            if load_inst_info:
                inst_labels = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_name', '%s.json'%scan_id)))
                inst_labels = [self.cat2int[i] for i in inst_labels]
                inst_locs = np.load(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_loc', '%s.npy'%scan_id))
                inst_colors = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_gmm_color', '%s.json'%scan_id)))
                inst_colors = [np.concatenate(
                    [np.array(x['weights'])[:, None], np.array(x['means'])],
                    axis=1
                ).astype(np.float32) for x in inst_colors]
                scans[scan_id] = {
                    'inst_labels': inst_labels, # (n_obj, )
                    'inst_locs': inst_locs,     # (n_obj, 6) center xyz, whl
                    'inst_colors': inst_colors, # (n_obj, 3x4) cluster * (weight, mean rgb)
                }
            else:
                scans[scan_id] = {}
                
            # load pcd data  # CHANGE FOR FEATURES
            pcd_data = torch.load(os.path.join(SCAN_FAMILY_BASE, "scan_data", "pcd_with_features_aligned", '%s.pth'% scan_id), weights_only=False)
            points, colors, features, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2], pcd_data[-1]
            instance_labels = instance_labels.astype(int)

            # pcd_data = torch.load(os.path.join(SCAN_FAMILY_BASE, "scan_data", "pcd_with_global_alignment", '%s.pth'% scan_id), weights_only=False)
            # points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
            
            # non_nan_inds = np.where(~np.isnan(points[:, 0]))[0]
            # points = points[non_nan_inds]
            # colors = colors[non_nan_inds]
            # features = features[non_nan_inds]
            # instance_labels = instance_labels[non_nan_inds]

            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors, features], 1)
            # pcds = np.concatenate([points, colors], 1)
            # convert to gt object
            if load_inst_info:
                obj_pcds = []
                for i in range(instance_labels.max() + 1):
                    mask = instance_labels == i     # time consuming
                    if mask.any():
                        obj_pcds.append(pcds[mask])      # (k_i, 14)
                    else:
                        # Pad with a single zero vector so shape is (1, 14)
                        missing_obj_count += 1
                        obj_pcds.append(np.zeros((1, pcds.shape[1]), dtype=pcds.dtype))    
                scans[scan_id]['pcds'] = obj_pcds          
                # calculate box for matching
                obj_center = []
                obj_box_size = []
                for i in range(len(obj_pcds)):
                    c, b = convert_pc_to_box(obj_pcds[i])
                    obj_center.append(c)
                    obj_box_size.append(b)
                scans[scan_id]['obj_center'] = obj_center
                scans[scan_id]['obj_box_size'] = obj_box_size
            
            # load mask
            if pc_type == 'pred':
                '''
                obj_mask_path = os.path.join(os.path.join(SCAN_FAMILY_BASE, 'mask'), str(scan_id) + ".mask" + ".npy")
                obj_label_path = os.path.join(os.path.join(SCAN_FAMILY_BASE, 'mask'), str(scan_id) + ".label" + ".npy")
                obj_pcds = []
                obj_mask = np.load(obj_mask_path)
                obj_labels = np.load(obj_label_path)
                obj_labels = [self.label_converter.nyu40id_to_id[int(l)] for l in obj_labels]
                '''
                obj_mask_path = os.path.join(MASK_BASE, str(scan_id) + ".mask" + ".npz")
                obj_label_path = os.path.join(MASK_BASE, str(scan_id) + ".label" + ".npy")
                obj_pcds = []
                obj_mask = np.array(sparse.load_npz(obj_mask_path).todense())[:50, :]
                obj_labels = np.load(obj_label_path)[:50]
                for i in range(obj_mask.shape[0]):
                    mask = obj_mask[i]
                    if pcds[mask == 1, :].shape[0] > 0:
                        obj_pcds.append(pcds[mask == 1, :])
                scans[scan_id]['pcds_pred'] = obj_pcds
                scans[scan_id]['inst_labels_pred'] = obj_labels[:len(obj_pcds)]
                # calculate box for pred
                obj_center_pred = []
                obj_box_size_pred = []
                for i in range(len(obj_pcds)):
                    c, b = convert_pc_to_box(obj_pcds[i])
                    obj_center_pred.append(c)
                    obj_box_size_pred.append(b)
                scans[scan_id]['obj_center_pred'] = obj_center_pred
                scans[scan_id]['obj_box_size_pred'] = obj_box_size_pred
        
        print(f"missing obj count: {missing_obj_count}")
        print("finish loading scannet data")
        return scans

    def load_rscan(self):
        scans = {}
        missing_obj_count = 0
        # attribute
        # inst_labels, inst_locs, inst_colors, pcds, / pcds_pred, inst_labels_pred
        folder_path = os.path.join(SCAN_FAMILY_BASE, '3rscan', 'feature_pcds_aligned_reoriented')
        for file_name in tqdm(os.listdir(folder_path)):
            scan_id = file_name.split('.')[0]
            file_path = os.path.join(folder_path, file_name)
            pcd_data = torch.load(file_path, weights_only=False)
            points, colors, features, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2], pcd_data[-1]
            points = points.cpu().numpy()
            instance_labels = instance_labels.astype(int)
            inst2label_path = os.path.join(SCAN_FAMILY_BASE, '3rscan', 'instance_id_to_label', f'{file_name}')
            inst_to_label = torch.load(inst2label_path)
                
            # load pcd data  # CHANGE FOR FEATURES
            # pcd_data = torch.load(os.path.join(SCAN_FAMILY_BASE, "scan_data", "pcd_with_features", '%s.pth'% scan_id), weights_only=False)
            # points, colors, features, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2], pcd_data[-1]
            # pcd_data = torch.load(file_path, weights_only=False)
            # points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]

            colors = colors / 127.5 - 1
            # pcds = np.concatenate([points, colors, features], 1)
            pcds = np.concatenate([points, colors, features], 1)

            # inst_labels = []
            # for i in range(instance_labels.max() + 1):
            #     if i in inst_to_label.keys():
            #         inst_labels.append(self.cat2int[inst_to_label[i]])
            #     else:
            #         inst_labels.append(-100)
            #         # tqdm.write(f"Warning: {i} not in inst_to_label")

            scans[scan_id] = {
                'inst_labels': {k:self.cat2int[v] for k, v in inst_to_label}, # (n_obj, )
            }

            # convert to gt object
            obj_pcds = {}
            for i in set(instance_labels):
                mask = instance_labels == i     # time consuming
                if mask.any():
                    obj_pcds[i] = pcds[mask]      # (k_i, 14)
                else:
                    # Pad with a single zero vector so shape is (1, 14)
                    missing_obj_count += 1
                    obj_pcds[i] = np.zeros((1, pcds.shape[1]), dtype=pcds.dtype)  
            scans[scan_id]['pcds'] = obj_pcds          
            # calculate box for matching
            # obj_center = []
            # obj_box_size = []
            # for i in range(len(obj_pcds)):
            #     c, b = convert_pc_to_box(obj_pcds[i])
            #     obj_center.append(c)
            #     obj_box_size.append(b)
            # scans[scan_id]['obj_center'] = obj_center
            # scans[scan_id]['obj_box_size'] = obj_box_size
 
        print(f"missing obj count: {missing_obj_count}")
        print("finish loading 3rscan data")
        return scans
class DataAugmentationMixin(object):
    def __init__(self):
        pass
        
    def build_rotate_mat(self):
        theta_idx = np.random.randint(len(ROTATE_ANGLES))
        theta = ROTATE_ANGLES[theta_idx]
        if (theta is not None) and (theta != 0) and (self.split == 'train'):
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            rot_matrix = None
        return rot_matrix
    
    def random_flip(self, point_cloud, p):
        ret_point_cloud = point_cloud.copy()
        if np.random.rand() < p:
            ret_point_cloud[:, 0] = point_cloud[:, 0]
        if np.random.rand() < p:
            ret_point_cloud[:, 1] = point_cloud[:, 1]
        return ret_point_cloud
    
    def set_color_to_zero(self, point_cloud, p):
        ret_point_cloud = point_cloud.copy()
        if np.random.rand() < p:
            ret_point_cloud[:, 3:6] = 0
        return ret_point_cloud
    
    def random_jitter(self, point_cloud):
        noise_sigma = 0.01
        noise_clip = 0.05
        jitter = np.clip(noise_sigma * np.random.randn(point_cloud.shape[0], 3), -noise_clip, noise_clip)
        ret_point_cloud = point_cloud.copy()
        ret_point_cloud[:, 0:3] += jitter
        return ret_point_cloud