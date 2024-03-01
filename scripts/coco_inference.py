import os
import sys
import operator
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import clip
import torch
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

import pickle
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

class COCOInference:
    def __init__(self, clip_model, checkpoint):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.categories = self.load_categories()
        self.data = self.load_data()
        self.model, self.preprocess = self.load_model(clip_model, checkpoint)
    
    def load_data(self):
        with open("./image_list/coco_test_images.pkl", "rb") as f:
            all_data = pickle.load(f)
        
        return all_data

    def load_categories(self, category_path = "./keywords/coco_categories.pkl"):
        with open(category_path, "rb") as f:
            categories = pickle.load(f)

        return categories
        
    def load_model(self, clip_model, checkpoint):
        model, preprocess = clip.load(clip_model, device = self.device)
        model.float()
        model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
        model.eval()
        
        return model, preprocess
    
    def eval_metrics(self, results):
        ### Precision, Recall, F1 Scores
        y_pred = np.zeros((len(results), len(self.categories)))
        y_true = np.zeros((len(results), len(self.categories)))
        coco_categories = list(self.categories.values())

        all_preds_ids = []
        for i, (_, result) in enumerate(results.items()):
            temp_labels = result[1]
            # temp_preds = list(result[0].keys())
            temp_preds = self.top_remove_softmax(result[0], threshold = 0.75)
            # temp_test = torch.Tensor(list(result[0].values()))

            for label in temp_labels:
                y_true[i, coco_categories.index(label)] = 1
                # if label in temp_preds:
                #     y_pred[i, coco_categories.index(label)] = 1
                #     all_preds_ids.append(coco_categories.index(label))

            for pred in temp_preds:
                # if result[0][pred] >= 18.5:
                y_pred[i, coco_categories.index(pred)] = 1
                all_preds_ids.append(coco_categories.index(pred))

        print(classification_report(y_true, y_pred, target_names = list(self.categories.values()), labels = np.unique(all_preds_ids), digits = 4))
    
    def top_remove_softmax(self, predictions, threshold = 0.5):
        temp_keywords = list(predictions.keys())
        temp_values = torch.Tensor(list(predictions.values())).detach()

        predicted = []
        for i in range(len(temp_values)):
            temp_value = temp_values[i].exp()
            temp_sum = torch.sum(temp_values[i:].exp())
            temp_prob = temp_value / temp_sum
            if temp_prob.item() >= threshold:
                predicted.append(i)
            else:
                break
        
        predicted_keywords = []
        for predict in predicted:
            predicted_keywords.append(temp_keywords[predict])

        return predicted_keywords
    
    def run_similarity_inference(self):
        categories = list(self.categories.values())
        prompts = clip.tokenize([f"a photo of a {c}" for c in categories]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(prompts)
        text_features /= text_features.norm(dim = -1, keepdim = True)

        results = {}
        for s_data in tqdm(self.data):
            image = s_data[0]
            labels = s_data[1]
    
            image_input = self.preprocess(Image.open(image).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim = -1, keepdim = True)

            similarity = (100.0 * image_features @ text_features.T)
            values, indices = similarity[0].topk(len(categories))

            clsf_results = {}
            for value, index in zip(values, indices):
                clsf_results[categories[index]] = value.item()
            
            results[image] = [clsf_results, labels]

        self.eval_metrics(results)

        with open("./results/coco_results.pkl", "wb") as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    inference = COCOInference(clip_model = "ViT-L/14", checkpoint = "./checkpoints/ViT-L14_COCO/interpolated/best_model_alpha_0.8_interpolated.pt")
    # inference.run_similarity_inference()

    with open("./results/coco_results.pkl", "rb") as f:
        results = pickle.load(f)
    inference.eval_metrics(results)