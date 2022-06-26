import os
import json

from torch.utils.data import Dataset

from PIL import Image

from data.utils import pre_caption_minimum, wait_for_file

from collections import defaultdict
import numpy as np
import random
import torch

from glob import glob
import av
import decord
from decord import VideoReader

from torchvision import transforms
import visual_genome.local as vg

# from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
# from sklearn.cluster import KMeans

import copy


class visual_genome_dataset(Dataset):
    def __init__(self, transform, config, max_words=64):
        '''
        config:
            video_roots (list(string): list of Root directory of videos (e.g. msrvtt_ret/videos/)
            train_ann_jsons (list(string)): ["dataset1.json",...]
            video_formats (list(string)): ["mp4",...]
            num_frm_train (string): number of frames to be sampled
            frm_sampling_strategy (string): ["uniform","nlvl_uniform","nlvl_rand","rand","headtail"]
            height=None, 
            width=None, 
            start_time=None,
            end_time=None, 
            fps=-1
        '''
        ids = vg.get_all_image_data()
        print ("We got an image with id: ",ids)
        self.config = config

        ann_jsons = config['train_ann_jsons'] # contains videoids and its corresponding text
        video_roots = config['video_roots']
        video_formats = config['video_formats']

        if isinstance(ann_jsons,str):
            ann_jsons = [ann_jsons]
            video_roots = [video_roots]
            video_formats = [video_formats]
    
        if ann_jsons == 'database':
            self.movie_db = MOVIE_DB()
            self.db = self.movie_db.db
             
        assert len(ann_jsons) == len(video_roots) == len(video_formats)
        
        # load annotation
        self.annotation = {}
        skipped_count = 0
        for i in range(len(ann_jsons)):
            #Insert database related code here
            #ann = json.load(open(ann_jsons[i]))
            video_dir = video_roots[i]
            video_fmt = video_formats[i]
            #if isinstance(ann, list):
                #for obj in ann:
            video_id = "video6250" #obj['video_id']
            video_path = os.path.join(video_dir,f'{video_id}.{video_fmt}')
            if not os.path.exists(video_path):
                print(f'ERROR: video file not found, skipped:{video_path}')
                skipped_count += 1
                continue
            # assume a list of text
            if video_id not in self.annotation:
                self.annotation[video_id] = {'video': video_path, 'caption':[]}
            #assert isinstance(obj['texts'],list)
            self.annotation[video_id]['caption'] = "demi caption" #obj['texts']
                    
            # elif isinstance(ann, dict):
            #     # assume keys are video ids
            #     for video_id, texts in ann.items():
            #         video_path = os.path.join(video_dir,f'{video_id}.{video_fmt}')
            #         if not os.path.exists(video_path):
            #             print(f'ERROR: video file not found, skipped:{video_path}')
            #             skipped_count += 1
            #             continue
            #         assert isinstance(texts,list)
            #         self.annotation[video_id] = {'video': video_path, 'caption':texts}
        
        self.annotation = [value for key,value in self.annotation.items()]
        print('num of video skipped:', skipped_count )
        print('num of video considering:', len(self.annotation))

        self.transform = transform
        # add ToPILImage: to let the extracted frames from decord be able to use the training transform 
        self.transform.transforms.insert(0, transforms.ToPILImage())

        self.max_words = max_words

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]

        video_path = ann["video"]
        caption = ann["caption"]
        
        # try loading video
        for _ in range(3):
            raw_sample_frms = self._load_video_from_path_decord(video_path)
            if raw_sample_frms is not None:
                break
        # return None if cannot load
        if raw_sample_frms is None:
            return None, None
        processed_frms = [self.transform(frm) for frm in raw_sample_frms]
        if not isinstance(processed_frms[0],Image.Image):
            processed_frms = torch.stack(processed_frms) # [num_frm, c, h, w]
        
        if 'timesformer' in self.config["vit"]:
            processed_frms = processed_frms.permute(1,0,2,3)
        
        return processed_frms, caption

    def _load_video_from_path_decord(self, video_path):
        frm_sampling_strategy=self.config['frm_sampling_strategy']
        num_frm=self.config['num_frm_train']
        height=self.config['height']
        width=self.config['width']
        start_time=self.config['start_time']
        end_time=self.config['end_time']
        fps=self.config['fps']

        try:
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)

            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
            else:
                start_idx, end_idx = 0, vlen

            if frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / num_frm, dtype=int)
            elif frm_sampling_strategy == 'nlvl_uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / num_frm).astype(int)
            elif frm_sampling_strategy == 'nlvl_rand':
                frame_indices = np.arange(start_idx, end_idx, vlen / num_frm).astype(int)

                # generate some random perturbations
                strides = [frame_indices[i] - frame_indices[i-1] for i in range(1, len(frame_indices))] + [vlen - frame_indices[-1]]
                pertube = np.array([np.random.randint(0, stride) for stride in strides])

                frame_indices = frame_indices + pertube

            elif frm_sampling_strategy == 'rand':
                frame_indices = sorted(random.sample(range(vlen), num_frm))
            elif frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(range(vlen // 2), num_frm // 2))
                frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            elif frm_sampling_strategy == 'clip-kmeans':
                frame_indices = self._CLIP_selection(vr, num_frm)
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices).detach().cpu().numpy() # (num_frm, H, W, C)
            # raw_sample_frms = vr.get_batch(frame_indices).asnumpy() # (num_frm, H, W, C)

        except Exception as e:
            print(e)
            return None
        # raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2) # torch tensor
        # raw_sample_frms = np.transpose(raw_sample_frms, (0, 3, 1, 2)) # numpy
        return raw_sample_frms
