import os
import json

from torch.utils.data import Dataset

from PIL import Image

from data.utils import pre_caption_minimum, wait_for_file

from collections import defaultdict
import numpy as np
import torch

from glob import glob
import ruamel.yaml as yaml
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode



# from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
# from sklearn.cluster import KMeans

import copy


class visual_genome_dataset(Dataset):
    def __init__(self, mod = 'blip', max_words=64):
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
        config = yaml.load(open('configs/pipeline_config_vg_test_nebula_toc.yaml', 'r'), Loader=yaml.Loader)
        if mod == 'blip':
            normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))       
            transform_ = transforms.Compose([
                transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
                ])
        elif mod == 'clip':
            transform_ = transforms.Compose([])

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
                    
        self.annotation = [value for key,value in self.annotation.items()]
        print('num of images skipped:', skipped_count )
        print('num of images considering:', len(self.annotation))

        self.transform = transform_
        # add ToPILImage: to let the extracted frames from decord be able to use the training transform 
        #self.transform.transforms.insert(0, transforms.ToPILImage())
        self.config = config
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
        croped_images.append(image)
        names.append(["full image"])
        #print("OBJECTS ", len(ann['objects']))
        for i, visual_objects in enumerate(ann['objects']):
            h = visual_objects['h']
            w = visual_objects['w']
            y = visual_objects['y']
            x = visual_objects['x']
            x,y,w,h = self.bbox_xywh_to_xyxy((x,y,w,h))
           
            crop_image = image.crop((x,y,w,h))
            width, height = crop_image.size
            if width > 30 and height > 30:
                #print(width, height)
                croped_images.append(crop_image)
                names.append(visual_objects['names'])
        
        processed_frms = [self.transform(frm) for frm in croped_images]
        if not isinstance(processed_frms[0],Image.Image):
            processed_frms = torch.stack(processed_frms)
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
        
        