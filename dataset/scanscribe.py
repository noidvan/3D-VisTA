import collections
import json
from tqdm import tqdm
import multiprocessing as mp
import os
import random
from copy import deepcopy

import jsonlines
import numpy as np
import torch
from dataset.dataset_mixin import DataAugmentationMixin, LoadScannetMixin
from dataset.path_config import SCAN_FAMILY_BASE
from torch.utils.data import Dataset
from utils.eval_helper import (construct_bbox_corners, convert_pc_to_box,
                               eval_ref_one_sample, is_explicitly_view_dependent)
from utils.label_utils import LabelConverter
from collections import Counter

class ScanScribeDataset(Dataset, LoadScannetMixin, DataAugmentationMixin):
    def __init__(self, split='train', anno_type='nr3d', max_obj_len=60, num_points=1024, pc_type='gt', sem_type='607', filter_lang=False, sr3d_plus_aug=False):
        # make sure all input params is valid
        # use ground truth for training
        # test can be both ground truth and non-ground truth
        assert pc_type in ['gt', 'pred']
        assert sem_type in ['607']
        assert split in ['train', 'val', 'test']
        assert anno_type in ['nr3d', 'sr3d']
        if split == 'train':
            pc_type = 'gt'

        # load category file
        self.int2cat = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2-labels.combined.tsv"))
        
        # load split file and create data storage
        split_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/splits/scannetv2_'+ split + ".txt")
        split_scan_ids = set([x.strip() for x in open(split_file, 'r')])
        self.scan_ids = set() # scan ids in data
        self.data = [] # scanrefer data

        # load Referit3D - NR3D
        nr3d_count = 0
        sr3d_count = 0
        anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/refer/' + anno_type + '.jsonl')
        with jsonlines.open(anno_file, 'r') as f:
            for item in f:
                if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                    item['object_ids'] =  [int(item['target_id'])]
                    item['object_names'] = [item['instance_type']]
                    self.scan_ids.add(item['scan_id'])
                    self.data.append(item)
                    nr3d_count += 1
        # load Referit3D - SR3D
        if sr3d_plus_aug and split == 'train':
            anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/refer/' + 'sr3d+' + '.jsonl')
            with jsonlines.open(anno_file, 'r') as f:
                for item in f:
                    if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                        item['object_ids'] =  [int(item['target_id'])]
                        item['object_names'] = [item['instance_type']]
                        self.scan_ids.add(item['scan_id'])
                        self.data.append(item)
                        sr3d_count += 1
        
        # load ScanRefer
        scanrefer_count = 0
        anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/refer/scanrefer.jsonl')
        with jsonlines.open(anno_file, 'r') as f:
            for item in f:
                if item['scan_id'] in split_scan_ids:
                    # Convert labels
                    # item['instance_type'] = self.int2cat[self.label_converter.id_to_scannetid[self.cat2int[item['instance_type']]]]
                    item['object_ids'] =  [int(item['target_id'])]
                    item['object_names'] = [item['instance_type']]
                    self.scan_ids.add(item['scan_id'])
                    self.data.append(item)
                    scanrefer_count += 1

        # load ScanQA
        scanqa_count = 0
        anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/qa/ScanQA_v1.0_' + split + ".json")
        json_data = json.load(open(anno_file, 'r'))
        for item in json_data:
            if item['scene_id'] in split_scan_ids:
                # Convert labels
                item['scan_id'] = item['scene_id']
                item['utterance'] = item['question']
                item['item_id'] = item['question_id']
                self.scan_ids.add(item['scan_id'])
                self.data.append(item)
                scanqa_count += 1

        # load ScanScribe
        scanscribe_template_count = 0
        scanscribe_gpt_count = 0
        if split == 'train':
            anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/scanscribe/template_gen_language.json')
            json_data = json.load(open(anno_file, 'r'))
            for item in json_data:
                item['utterance'] = item['sentence']
                # self.scan_ids.add(item['scan_id'])
                self.data.append(item)
                scanscribe_template_count += 1
            
            anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/scanscribe/gpt_gen_language.json')
            json_data = json.load(open(anno_file, 'r'))
            for i in range(15):
                for item in json_data:
                    item['utterance'] = item['sentence']
                    # self.scan_ids.add(item['scan_id'])
                    self.data.append(item)
                    scanscribe_gpt_count += 1

        print("Loaded %d ScanRefer, %d NR3D, %d SR3D, %d ScanQA, %d ScanScribe template, %d ScanScribe GPT." % (scanrefer_count, nr3d_count, sr3d_count, scanqa_count, scanscribe_template_count, scanscribe_gpt_count))
        # fill parameters
        self.split = split
        self.max_obj_len = max_obj_len - 1
        self.num_points = num_points
        self.pc_type = pc_type
        self.sem_type = sem_type
        self.filter_lang = filter_lang
        self.anno_type = anno_type
        
        # load scans
        self.scans = self.load_scannet(self.scan_ids, self.pc_type, self.split != 'test')
        if self.split == 'train':
            rscan = self.load_rscan()
            self.scans.update(rscan)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # load scanrefer
        item = self.data[idx]
        # item_id = item['item_id']
        scan_id =  item['scan_id']
        # tgt_object_id = int(item['target_id'])
        # tgt_object_name = item['instance_type']
        if 'object_ids' in item:
            tgt_object_id_list = item['object_ids']
            tgt_object_name_list = item['object_names']
        sentence = item['utterance']
        # is_view_dependent = is_explicitly_view_dependent(item['tokens'])
        
        # load pcds and labels
        obj_pcds = deepcopy(self.scans[scan_id]['pcds']) # N, 6
        obj_labels = deepcopy(self.scans[scan_id]['inst_labels']) # N
            
        # filter out background
        if isinstance(obj_pcds, list):
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
        elif isinstance(obj_pcds, dict):
            selected_obj_idxs = [k for k, v in obj_labels.items() if (self.int2cat[v] not in ['wall', 'floor', 'ceiling'])]
        else:
            raise ValueError("obj_pcds should be list or dict")
        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]
              
        # build tgt object id and box 
        if self.max_obj_len < len(obj_labels):
            if 'object_ids' in item:
                tgt_object_id_list = [selected_obj_idxs.index(x) for x in tgt_object_id_list]
                tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]
                for i in range(len(tgt_object_label_list)):
                    assert(self.int2cat[tgt_object_label_list[i]] == tgt_object_name_list[i])
                selected_obj_idxs = tgt_object_id_list.copy()
                remained_obj_idx = []
                for kobj, klabel in enumerate(obj_labels):
                    if kobj not in  tgt_object_id_list:
                        if klabel in tgt_object_label_list:
                            selected_obj_idxs.append(kobj)
                        else:
                            remained_obj_idx.append(kobj)
                    if len(selected_obj_idxs) == self.max_obj_len:
                        break
                if len(selected_obj_idxs) < self.max_obj_len:
                    random.shuffle(remained_obj_idx)
                    selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            else:
                # randomly select objects
                selected_obj_idxs = list(range(len(selected_obj_idxs)))
                random.shuffle(selected_obj_idxs)
                selected_obj_idxs = selected_obj_idxs[:self.max_obj_len]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            assert len(obj_pcds) == self.max_obj_len
        
        # rebuild tgt_object_id
            
        # rotate obj
        rot_matrix = self.build_rotate_mat()
        
        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for obj_pcd in obj_pcds:
            # build locs
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            # build box
            obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
            obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))
            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)
            
        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)
        
        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_fts.shape[0] == obj_locs.shape[0]
        
        data_dict = {
            "scan_id": scan_id,
            "sentence": sentence,
            "obj_fts": obj_fts, # N, 6
            "obj_locs": obj_locs, # N, 3
            "obj_labels": obj_labels, # N,
            "obj_boxes": obj_boxes, # N, 6 
        }
    
        return data_dict