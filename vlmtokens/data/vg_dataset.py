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
        self.config = config

        ann_jsons = config['vg_objects'] # contains videoids and its corresponding text
        image_roots = config['image_roots']
        image_formats = config['image_formats']

        if isinstance(ann_jsons,str):
            ann_jsons = [ann_jsons]
            image_roots = [image_roots]
            image_formats = [image_formats]
             
        assert len(ann_jsons) == len(image_roots) == len(image_formats)
        
        # load annotation
        self.annotation = {}
        skipped_count = 0
        for i in range(len(ann_jsons)):
            #Insert database related code here
            ann = json.load(open(ann_jsons[i]))
            image_dir = image_roots[i]
            image_fmt = image_formats[i]
            if isinstance(ann, list):
                for obj in ann:
                    image_id = obj['image_id']
                    image_path = os.path.join(image_dir,f'{image_id}.{image_fmt}')
                    if not os.path.exists(image_path):
                        #print(f'ERROR: image file not found, skipped:{image_path}')
                        skipped_count += 1
                        continue
                    # assume a list of text
                    if image_id not in self.annotation:
                        self.annotation[image_id] = {'image': image_path, 'caption':[]}
                    #assert isinstance(obj['texts'],list)
                    self.annotation[image_id]['objects'] = obj['objects']
                    
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
        print('num of images skipped:', skipped_count )
        print('num of images considering:', len(self.annotation))

        self.transform = transform
        # add ToPILImage: to let the extracted frames from decord be able to use the training transform 
        #self.transform.transforms.insert(0, transforms.ToPILImage())

        self.max_words = max_words

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
            
        ann = self.annotation[index]
        #print(ann)
        print(ann['image'])
        image_path = ann['image']        
        image = Image.open(image_path).convert('RGB')   
        #image = self.transform(image)
        names = []
        croped_images = []
        for visual_objects in ann['objects']:
            h = visual_objects['h']
            w = visual_objects['w']
            y = visual_objects['y']
            x = visual_objects['x']
            x,y,w,h = self.bbox_xywh_to_xyxy((x,y,w,h))
           
            
            crop_image = image.crop((x,y,w,h))
            width, height = crop_image.size
            if width > 30 and height > 30:
                #print(width, height)
                image.save(str(index) + ".jpg")
                croped_images.append(image)
                #crop_image.save(visual_objects['names'][0] + "_" + str(index) + ".jpg")
                croped_images.append(crop_image)
                names.append(visual_objects['names'])
        processed_frms = [self.transform(frm) for frm in croped_images]


        return processed_frms, names
    
    def bbox_xywh_to_xyxy(self, xywh):
        """Convert bounding boxes from format (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)

        Parameters
        ----------
        xywh : list, tuple or numpy.ndarray
            The bbox in format (x, y, w, h).
            If numpy.ndarray is provided, we expect multiple bounding boxes with
            shape `(N, 4)`.

        Returns
        -------
        tuple or numpy.ndarray
            The converted bboxes in format (xmin, ymin, xmax, ymax).
            If input is numpy.ndarray, return is numpy.ndarray correspondingly.

        """
        if isinstance(xywh, (tuple, list)):
            if not len(xywh) == 4:
                raise IndexError(
                    "Bounding boxes must have 4 elements, given {}".format(len(xywh)))
            w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
            return xywh[0], xywh[1], xywh[0] + w, xywh[1] + h
        elif isinstance(xywh, np.ndarray):
            if not xywh.size % 4 == 0:
                raise IndexError(
                    "Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
            xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
            return xyxy
        else:
            raise TypeError(
                'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh))) 
        
        # ann = self.annotation[index]

        # image_path = ann["video"]
        # caption = ann["caption"]
        
        # # try loading video
        # for _ in range(3):
        #     raw_sample_frms = self._load_vg_image_from_path_decord(image_path)
        #     if raw_sample_frms is not None:
        #         break
        # # return None if cannot load
        # if raw_sample_frms is None:
        #     return None, None
        # processed_frms = [self.transform(frm) for frm in raw_sample_frms]
        # if not isinstance(processed_frms[0],Image.Image):
        #     processed_frms = torch.stack(processed_frms) # [num_frm, c, h, w]
        
        # if 'timesformer' in self.config["vit"]:
        #     processed_frms = processed_frms.permute(1,0,2,3)
        
        # return processed_frms, caption

   