import clip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# from pycocotools.coco import COCO

from PIL import Image
import pickle
import os


### pycocotools로 데이터를 호출하는 것이 생각보다 시간이 걸림
### train과 test 셋을 추가적으로 분리해야 하기 때문에 사전에 이미지 경로와 라벨을 별도로 처리하여 pkl형식 파일로 저장함
class COCODataset(Dataset) :
    def __init__(self, data_path, transform):
        super(COCODataset, self).__init__()
        self.data_path = data_path
        self.transform = transform

        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        image = Image.open(image).convert("RGB")
        image = self.transform(image)

        texts = self.load_annotations(self.data[idx][1])

        return image, texts

    def load_data(self):
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        return data
    
    def load_annotations(self, categories):
        # with open("./keywords/coco_categories.pkl", 'rb') as f:
        #     coco_cats = pickle.load(f)

        # categories = list(set(categories))
        # if 80 in categories:
        #     categories.remove(80)
        
        # if 89 in categories:
        #     categories.remove(89)
        
        categories = categories[:15]
        texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories])
        texts = self.add_paddings(texts)

        # categories = categories[0] ### Change Label Lengths
        # texts = clip.tokenize(f"a photo of {categories}").squeeze(0)

        return texts

    def add_paddings(self, text_labels):
        if len(text_labels) < 15:
            padding_size = 15 - len(text_labels)
            pads = torch.zeros(padding_size, 77).type(torch.LongTensor) # CLIP Embeddings have 77 lengths
            text_labels = torch.cat([text_labels, pads])

        return text_labels