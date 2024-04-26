import torch
import clip
import pickle

from sklearn.metrics import classification_report

from PIL import Image
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class CLIPInference:
    def __init__(self, clip_model, checkpoint):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.categories = self.load_categories()
        self.model, self.preprocess = self.load_model(clip_model, checkpoint)
        self.data = self.load_data()
        
    def load_categories(self, category_path = "./keywords/coco_categories.pkl"):
        with open(category_path, "rb") as f:
            categories = pickle.load(f)

        return categories
    
    def load_data(self):
        with open("./image_list/coco_test_images.pkl", "rb") as f:
            all_data = pickle.load(f)
        
        return all_data
    
    def load_model(self, clip_model, checkpoint):
        model, preprocess = clip.load(clip_model, device = self.device)
        model.float() # CLIP uses float16, convert to float
        model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
        
        return model, preprocess
    
    def eval_metrics(self, results):
        ### Precision, Recall, F1 Scores
        y_pred = torch.zeros(len(results), len(self.categories))
        y_true = torch.zeros(len(results), len(self.categories))
        coco_categories = list(self.categories.values())

        for i, (image, result) in enumerate(results.items()):
            temp_labels = result[1]
            temp_preds = list(result[0].keys())

            y_true[i, coco_categories.index(temp_labels)] = 1

            if result[0][temp_preds[0]] >= 0.5:
            # if temp_labels in temp_preds:
                y_pred[i, coco_categories.index(temp_preds[0])] = 1

        print(classification_report(y_true, y_pred, target_names = list(self.categories.values())))

    ### 예측 결과에서 Softmax 확률이 가장 높은 하나만 이용해 평가 수행
    def run_inference(self):
        results = {}
        categories = list(self.categories.values())
        for s_data in tqdm(self.data):
            image_path = s_data[0]
            image_input = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
            
            labels = s_data[1][0] ### Single Label
            prompts = clip.tokenize([f"a photo of {c}" for c in categories]).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(prompts)
            image_features /= image_features.norm(dim = -1, keepdim = True)
            text_features /= text_features.norm(dim = -1, keepdim = True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim = -1)
            values, indices = similarity[0].topk(5)

            clsf_results = {}
            for value, idx in zip(values, indices):
                clsf_results[categories[idx]] = value.item()
            
            results[image_path] = [clsf_results, labels]

        self.eval_metrics(results)

if __name__ == "__main__":
    inference = CLIPInference(clip_model = "ViT-B/32", checkpoint = "./checkpoints/ViT-B32_COCO/best_model.pt")
    inference.run_inference()