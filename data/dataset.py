import torch
from torch.utils.data import Dataset
import clip

from PIL import Image
import pickle
import os

class ImageDataset(Dataset):
    def __init__(self, data_path, transform, label_root = "../car_data/removed_image_labels"):
        self.images = self.load_images(data_path)
        self.transform = transform
        self.label_root = label_root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_name = image_path.split('/')[-1]

        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)

        texts = self.get_labels(image_name)

        return img, texts
    
    def load_images(self, data_path):
        with open(data_path, 'rb') as f:
            image_data = pickle.load(f)
        
        return image_data
    
    def get_labels(self, image_name):
        if image_name.startswith("sketch"):
            image_name = image_name.split('sketch_')[1]
        
        image_name = os.path.splitext(image_name)[0]

        label_path = "%s/%s.txt" % (self.label_root, image_name)

        with open(label_path, 'r') as f:
            labels = eval(f.read())

        text_labels = []
        ### CLIP은 문장 형태로 타겟 클래스를 입력하기 때문에 키워드 형태의 타겟 클래스를 문장으로 변환
        for label in labels:
            temp_prompt = "a photo of car with %s design" % label
            prompt_label = clip.tokenize(temp_prompt)
            text_labels.append(prompt_label)
        
        text_labels = torch.cat(text_labels)
        ### 이미지 별로 라벨 수가 다르기 때문에 부족한 라벨은 0으로 채워줌
        text_labels = self.add_paddings(text_labels)
        
        return text_labels
    
    def add_paddings(self, text_labels):
        if len(text_labels) < 5:
            ### 최대 5개 라벨을 가짐
            padding_size = 5 - len(text_labels)
            pads = torch.zeros(padding_size, 77).type(torch.LongTensor) # CLIP Embeddings have 77 lengths
            text_labels = torch.cat([text_labels, pads])

        return text_labels