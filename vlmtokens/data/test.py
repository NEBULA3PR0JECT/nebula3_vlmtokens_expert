from vlmtokens.data.vg_dataset import visual_genome_dataset
from torchvision import transforms
import ruamel.yaml as yaml


video_dataset = visual_genome_dataset(max_words=64)
for i in range(len(video_dataset)):
    a,b = video_dataset[i]
    print(a)
    print(b)
