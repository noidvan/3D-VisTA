import numpy as np
import torch

from dataset.data_converter import *

class ScanFamilyDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_seq_length=80, max_obj_len=80):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_obj_len = max_obj_len
    
    def __len__(self):
        return len(self.dataset)
    
    def pad_tensors(self, tensors, lens=None, pad=0):
        try:
            assert tensors.shape[0] <= lens
        except:
            print(tensors.shape[0], lens)
            print(tensors.shape)
        if (tensors.shape[0] == lens):
            return tensors
        shape = list(tensors.shape)
        shape[0] = lens - shape[0]
        res = torch.ones(shape, dtype=tensors.dtype) * pad
        res = torch.cat((tensors, res), dim=0)
        return res
        
    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        sentence = data_dict['sentence']
        encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        # build object
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) # O
        data_dict['obj_fts'] = self.pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len, pad=1.0).float() # O, 1024, 6
        data_dict['obj_locs']= self.pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len, pad=0.0).float() # O, 3
        data_dict['obj_boxes']= self.pad_tensors(data_dict['obj_boxes'], lens=self.max_obj_len, pad=0.0).float() # O, 3
        data_dict['obj_labels'] = self.pad_tensors(data_dict['obj_labels'], lens=self.max_obj_len, pad=-100).long() # O
        # build sem mask, no mask
        data_dict['obj_sem_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs']))
        # build label for refer
        data_dict['tgt_object_label'] = data_dict['tgt_object_label'].long() # 1 or C
        data_dict['tgt_object_id'] = data_dict['tgt_object_id'].long() # 1 or O
        if len(data_dict['tgt_object_id']) > 1: # O, pad to max objet length
            data_dict['tgt_object_id'] = self.pad_tensors(data_dict['tgt_object_id'].long(), lens=self.max_obj_len, pad=0).long() # O
        # build target
        if data_dict.get('tgt_object_id_iou25') != None:
            data_dict['tgt_object_id_iou25'] = self.pad_tensors(data_dict['tgt_object_id_iou25'], lens=self.max_obj_len, pad=0).long()
        if data_dict.get('tgt_object_id_iou50') != None:
            data_dict['tgt_object_id_iou50'] = self.pad_tensors(data_dict['tgt_object_id_iou50'], lens=self.max_obj_len, pad=0).long()
        # build label for qa
        if "answer_label" in data_dict:
            data_dict['answer_label'] = data_dict['answer_label'].long() # N, C
        return data_dict

class MaskDatasetWrapper(torch.utils.data.Dataset):
    # def __init__(self, cfg, dataset, split="train"):
    def __init__(self, dataset, tokenizer, max_seq_length=80, max_obj_len=80, txt_mask_ratio=0.15, pc_mask_ratio=0.1, replace_ratio=0.3):
        # tokenizer, max_seq_length=80, max_obj_len=80,
        #  mask_strategy='random', txt_mask_ratio=0.15, pc_mask_ratio=0.1
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_obj_len = max_obj_len
        self.txt_mask_ratio = txt_mask_ratio
        self.pc_mask_ratio = pc_mask_ratio
        self.replace_ratio = replace_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]

        sentence = data_dict['sentence']
        replace_label = False

        # Decide whether to replace for scene text match
        if random.random() < self.replace_ratio:
            # Pick a different random index
            other_idx = random.randrange(len(self.dataset))
            while other_idx == idx:
                other_idx = random.randrange(len(self.dataset))

            # Swap the caption and mark as “replaced”
            sentence = self.dataset[other_idx]['sentence']
            replace_label  = True

        data_dict['replace']  = replace_label # 0 → match, 1 → mismatch

        encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L

        # mask txt
        masked_txt_ids, masked_lm_labels = random_word(data_dict['txt_ids'], data_dict['txt_masks'],
                                                       self.tokenizer, self.txt_mask_ratio)
        data_dict['txt_ids'] = masked_txt_ids
        data_dict['masked_lm_labels'] = masked_lm_labels
        # build object
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) # O
        data_dict['obj_fts'] = pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len, pad=1.0).float() # O, 1024, 6
        data_dict['obj_locs']= pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len, pad=0.0).float() # O, 3
        data_dict['obj_boxes']= pad_tensors(data_dict['obj_boxes'], lens=self.max_obj_len, pad=0.0).float() # O, 3
        data_dict['obj_labels'] = pad_tensors(data_dict['obj_labels'], lens=self.max_obj_len, pad=-100).long() # O
        # mask object, 0 means mask object, 1 means keep object
        if 'obj_fts' in data_dict.keys():
            obj_sem_masks = random_point_cloud(data_dict['obj_fts'], data_dict['obj_masks'],
                                            self.pc_mask_ratio)
            data_dict['obj_sem_masks'] = obj_sem_masks
        else:
            obj_sem_masks = []
            for i in range(self.max_obj_len):
                if i >= len(data_dict['obj_locs']):
                    obj_sem_masks.append(0)
                else:
                    prob = random.random()
                    if prob < self.pc_mask_ratio:
                        obj_sem_masks.append(0)
                    else:
                        obj_sem_masks.append(1)
            data_dict['obj_sem_masks'] = torch.tensor(obj_sem_masks).long()
        # data_dict['tgt_object_label'] = data_dict['tgt_object_label'].long() # 1 or C
        # data_dict['tgt_object_id'] = data_dict['tgt_object_id'].long() # 1 or O

        # if len(data_dict['tgt_object_id']) > 1: # O, pad to max objet length
        #     data_dict['tgt_object_id'] = self.pad_tensors(data_dict['tgt_object_id'].long(), lens=self.max_obj_len, pad=0).long() # O
        # # build target
        # if data_dict.get('tgt_object_id_iou25') != None:
        #     data_dict['tgt_object_id_iou25'] = self.pad_tensors(data_dict['tgt_object_id_iou25'], lens=self.max_obj_len, pad=0).long()
        # if data_dict.get('tgt_object_id_iou50') != None:
        #     data_dict['tgt_object_id_iou50'] = self.pad_tensors(data_dict['tgt_object_id_iou50'], lens=self.max_obj_len, pad=0).long()
        # # build label for qa
        # if "answer_label" in data_dict:
        #     data_dict['answer_label'] = data_dict['answer_label'].long() # N, C
        return data_dict



class CaptionDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, vocab, corpus, max_seq_length=80, max_obj_len=80, txt_mask_ratio=0.15, split='train'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.corpus = torch.load(corpus)
        self.max_seq_length = max_seq_length
        self.max_obj_len = max_obj_len
        self.txt_mask_ratio = txt_mask_ratio
        self.split = split
    
    def __len__(self):
        return len(self.dataset)
    
    def pad_tensors(self, tensors, lens=None, pad=0):
        try:
            assert tensors.shape[0] <= lens
        except:
            print(tensors.shape[0], lens)
            print(tensors.shape)
        if (tensors.shape[0] == lens):
            return tensors
        shape = list(tensors.shape)
        shape[0] = lens - shape[0]
        res = torch.ones(shape, dtype=tensors.dtype) * pad
        res = torch.cat((tensors, res), dim=0)
        return res
        
    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        sentence = data_dict['sentence']
        encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        # build object
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) # O
        data_dict['obj_fts'] = self.pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len, pad=1.0).float() # O, 1024, 6
        data_dict['obj_locs']= self.pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len, pad=0.0).float() # O, 3
        data_dict['obj_boxes']= self.pad_tensors(data_dict['obj_boxes'], lens=self.max_obj_len, pad=0.0).float() # O, 3
        data_dict['obj_labels'] = self.pad_tensors(data_dict['obj_labels'], lens=self.max_obj_len, pad=-100).long() # O
        # build sem mask, no mask
        data_dict['obj_sem_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs']))
        # build label for refer
        data_dict['tgt_object_label'] = data_dict['tgt_object_label'].long() # 1 or C
        data_dict['tgt_object_id'] = data_dict['tgt_object_id'].long() # 1 or O
        if len(data_dict['tgt_object_id']) > 1: # O, pad to max objet length
            data_dict['tgt_object_id'] = self.pad_tensors(data_dict['tgt_object_id'].long(), lens=self.max_obj_len, pad=0).long() # O
        # build target
        if data_dict.get('tgt_object_id_iou25') != None:
            data_dict['tgt_object_id_iou25'] = self.pad_tensors(data_dict['tgt_object_id_iou25'], lens=self.max_obj_len, pad=0).long()
        if data_dict.get('tgt_object_id_iou50') != None:
            data_dict['tgt_object_id_iou50'] = self.pad_tensors(data_dict['tgt_object_id_iou50'], lens=self.max_obj_len, pad=0).long()
        # build input output for caption
        if self.split == 'train':
            masked_txt_ids, masked_lm_labels = random_caption_word(data_dict['txt_ids'], data_dict['txt_masks'], self.tokenizer, self.vocab, self.txt_mask_ratio)
            data_dict['txt_ids'] = masked_txt_ids
            data_dict['masked_lm_labels'] = masked_lm_labels
        else:
            data_dict['gt_ids'] = data_dict['txt_ids'].clone()
            sentence = ""
            for i in range(self.max_seq_length - 2):
                sentence += '[MASK]'
            encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                        add_special_tokens=True, truncation=True,
                        padding='max_length', return_tensors="pt")
            data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
            data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        return data_dict
