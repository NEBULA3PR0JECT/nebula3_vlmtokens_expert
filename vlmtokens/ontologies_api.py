import json
import os
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from models.blip_retrieval import blip_retrieval
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import ruamel.yaml as yaml
from PIL import Image
import requests


DUMMY_IMAGE = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
EMBBDING_BATCH_LIMIT_TEXT = 512

class ONTOLOGY_API():
    def __init__(self, ontology='persons', vlm='clip', config_device='cuda'):
         ###
        OMIT_KEYWORDS = ['media player','video','playing video','audio','sound','taking video', 'water mark', 'water marked', 'watermark', 'watermarks', 'for sale in', 'sold from', 'stock', 'sold on','by viewers',
            'are provided by','are posted on','for more','tag with','stream from','viewed from','showing video of','are on at', 'shuttlecock', 'shutter', 'shutter is white', 'shutters have bones','tape is looped', 'bliss wants you','thumbnail','technique']
        ###
        
        device = torch.device(config_device)
        self.device = device
        config = yaml.load(open("/notebooks/nebula3_vlmtokens_expert/vlmtokens/configs/pipeline_config_msrvtt_test_nebula_toc.yaml", 'r'), Loader=yaml.Loader)
        print("Creating model")
        self.vlm = vlm
        if self.vlm == 'blip':
            model = blip_retrieval(pretrained=config['blip_model_visual_tokenization'], image_size=config['image_size'], vit=config['vit'], 
                                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                    queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
            self.model = model.to(device)
            self.processor = None
        elif self.vlm == 'clip':
            model_name = config['clip_model_visual_tokenization']
            print(f'loading {model_name}...')
            model = CLIPModel.from_pretrained(model_name)
            model.eval()
            self.model = model.to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        elif self.vlm == 'florence':
            #TODO_florencepyth  
            model = None
            model.eval()
            self.model = model.to(device)
            self.processor = None
        
        prompt_functions = self.get_prefix_prompt_functions(ontology)
        print('prompts:', prompt_functions)

        ''' load ontology '''
        print('using openimage objects and vg srl selected events')
        # object_json_path = f'shared_datasets/OpenImages/openimage_classes_600.json'
        object_json_path = f'visual_token_ontology/vg/openimage_classes_all_cleaned_fictional_characters.json'
        vg_object_json_path = f'visual_token_ontology/vg/objects_sorted_all.json'
        attribute_json_path = f'visual_token_ontology/vg/vg_original_attributes_synsets_keys_cleaned_remove_similar0.9.json'
        vg_attribute_json_path = f'visual_token_ontology/vg/attr_sorted_all.json'
        scene_json_path = f'visual_token_ontology/vg/place365_ontology.json'
        persons_json_path = f'visual_token_ontology/vg/persons_ontology.json'
        verb_json_path = f'visual_token_ontology/vg/vg_srl_selected_object_synsets_keys_remove_similar0.9.json'
        vg_verb_json_path = f'visual_token_ontology/vg/capable_of_sorted_all.json'
        indoor_json_path = f'visual_token_ontology/vg/indoor_ontology.json'
        if ontology == 'persons':
            ontology_texts = self.load_json(persons_json_path)
        elif ontology == 'attribytes':
            ontology_texts = self.load_json(attribute_json_path)
        elif ontology == 'vg_attribytes':
            ontology_texts = self.load_json(vg_attribute_json_path)
        elif ontology == 'objects':
            ontology_texts = self.load_json(object_json_path)
        elif ontology == 'vg_objects':
            ontology_texts = self.load_json(vg_object_json_path)
        elif ontology == 'verbs':
            ontology_texts = self.load_json(verb_json_path)
        elif ontology == 'vg_verbs':
            ontology_texts = self.load_json(vg_verb_json_path)
        else:
            print("Unknown Ontology")

        for key in OMIT_KEYWORDS:
            if key in ontology_texts: ontology_texts.remove(key)   
        self.config = config
        self.ontology_texts = ontology_texts
        self.prompt_functions = prompt_functions
        self.ontology = ontology
        self.vlm = vlm
    
    def load_json(self, json_path):
        return json.load(open(json_path))

    def get_prefix_prompt_functions(self, ontology):
        attribute_prompt = lambda x: f'A photo of {x}'
        scene_prompt = lambda x: f'A photo of {x}'
        verb_prompt = lambda x: f'A photo of {x}'
        object_prompt = lambda x: f'A photo of {x}'
        vg_attribute_prompt = lambda x: f'A photo of something or somebody {x}'
        persons_prompt = lambda x: f'A photo of {x}'
        scene_prompt = lambda x: f'A photo of {x}'
        vg_verb_prompt = lambda x: f'A photo of something capable of {x}'
        indoor_prompt = lambda x: f'A photo of {x}'
        return {
            'objects':object_prompt,
            'vg_objects':object_prompt,
            'attributes':attribute_prompt,
            'vg_attributes': vg_attribute_prompt,
            'scenes':scene_prompt,
            'persons': persons_prompt,
            'verbs':verb_prompt,
            'vg_verbs': vg_verb_prompt
            #'indoors': indoor_prompt
        }

    ### embedding functions ### 
    @torch.no_grad()
    def get_text_embeddings_clip(self, model, processor, texts, device):
        num_text = len(texts)
        text_bs = EMBBDING_BATCH_LIMIT_TEXT
        text_embeds = []  
        for i in range(0, num_text, text_bs):
            text = texts[i: min(num_text, i+text_bs)]
            inputs = processor(text=text, images=DUMMY_IMAGE, return_tensors="pt", padding=True, truncation = True).to(device)
            outputs = model(**inputs)
            txt_emb = outputs.text_embeds
            text_embeds.append(txt_emb)
        
        text_embeds = torch.cat(text_embeds,dim=0)
        return text_embeds, None, None

    @torch.no_grad()
    def get_text_embeddings_florence(self, model, processor, texts, device):
        # texts: a list of str
        num_text = len(texts)
        text_bs = EMBBDING_BATCH_LIMIT_TEXT
        text_embeds = []  
        for i in range(0, num_text, text_bs):
            text = texts[i: min(num_text, i+text_bs)]
            #TODO_florence
            txt_emb = None
            text_embeds.append(txt_emb)
        
        text_embeds = torch.cat(text_embeds,dim=0)
        return text_embeds, None, None

    @torch.no_grad()
    def get_text_embeddings_blip(self, model, texts, device):    
        num_text = len(texts)
        text_bs = EMBBDING_BATCH_LIMIT_TEXT
        text_ids = []
        text_embeds = []  
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i: min(num_text, i+text_bs)]
            text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
            text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
            text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
            text_embeds.append(text_embed)   
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)
        
        text_embeds = torch.cat(text_embeds,dim=0)
        text_ids = torch.cat(text_ids,dim=0)
        text_atts = torch.cat(text_atts,dim=0)
        text_ids[:,0] = model.tokenizer.enc_token_id
        return text_embeds, text_ids, text_atts


    def get_ontology(self):
        text_representations = {}
        texts = [self.prompt_functions[self.ontology](t) for t in self.ontology_texts]
        if self.vlm == 'blip':
            text_embeds, text_ids, text_atts = self.get_text_embeddings_blip(self.model, texts, self.device)
        elif self.vlm == 'clip':
            # text_ids, text_atts should be None
            text_embeds, text_ids, text_atts = self.get_text_embeddings_clip(self.model, self.processor, texts, self.device)
        elif self.vlm == 'florence':
            # text_ids, text_atts should be None
            text_embeds, text_ids, text_atts = self.get_text_embeddings_florence(self.model, self.processor, texts, self.device)
         
        text_representations[self.ontology] = {
            'text': texts,
            'text_embeds':text_embeds,
            'text_ids':text_ids,
            'text_atts':text_atts
        }
        return(text_representations)

    def get_vlmodel():
        return(self.model, self.processor, self.model_name)

test = ONTOLOGY_API('persons', 'blip')
print(test.get_ontology())