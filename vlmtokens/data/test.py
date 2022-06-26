from vlmtokens.data.vg_dataset import visual_genome_dataset
from torchvision import transforms
import ruamel.yaml as yaml

config = yaml.load(open('configs/pipeline_config_msrvtt_test_nebula_toc.yaml', 'r'), Loader=yaml.Loader)
mod = 'clip'
if mod == 'blip':
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))       
    transform_ = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
elif mod == 'clip':
    transform_ = transforms.Compose([])

video_dataset = visual_genome_dataset(transform_, config, max_words=64)
print(video_dataset)