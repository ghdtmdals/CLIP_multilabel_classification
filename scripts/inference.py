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
from utils.utils import read_files
from PIL import Image
from tqdm import tqdm
import random

class Inference:
    def __init__(self, data_path, clip_model, checkpoint):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.keywords = self.load_keywords()
        self.model, self.preprocess = self.load_model(clip_model, checkpoint)
        self.images = self.load_images(data_path)
    
    def load_images(self, data_path):
        with open(data_path, "rb") as f:
            images = pickle.load(f)
        
        images = random.sample(images, 50000)

        return images

    def load_keywords(self, keywords_path = "./keywords/all_keywords.pkl"):
        with open(keywords_path, "rb") as f:
            keywords = pickle.load(f)

        return keywords
        
    def load_model(self, clip_model, checkpoint):
        model, preprocess = clip.load(clip_model, device = self.device)
        model.float() # CLIP uses float16, convert to float
        model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
        
        return model, preprocess
    
    def get_labels(self, image):
        image_name = image.split("/")[-1]

        if image_name.startswith("sketch"):
            image_name = image_name.split('sketch_')[1]
        
        image_name = os.path.splitext(image_name)[0]

        label_path = "%s/%s.txt" % ("../car_data/removed_image_labels", image_name)

        with open(label_path, 'r') as f:
            labels = eval(f.read())

        return labels
    
    def eval_metrics(self, results):
        ### Precision, Recall, F1 Scores
        y_pred = np.zeros((len(results), len(self.keywords)))
        y_true = np.zeros((len(results), len(self.keywords)))

        all_preds_ids = []
        for i, (_, result) in enumerate(results.items()):
            temp_labels = result[1]
            # temp_preds = list(result[0].keys())
            temp_preds = self.top_remove_softmax(result[0], threshold = 0.6)

            for label in temp_labels:
                y_true[i, self.keywords.index(label)] = 1
                # if label in temp_preds:
                #     y_pred[i, coco_categories.index(label)] = 1
                #     all_preds_ids.append(coco_categories.index(label))
            
            for pred in temp_preds:
                # if result[0][pred] >= 20:
                y_pred[i, self.keywords.index(pred)] = 1
                all_preds_ids.append(self.keywords.index(pred))

        print(classification_report(y_true, y_pred, target_names = self.keywords, labels = np.unique(all_preds_ids), digits = 4))
    
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
    
    def run_inference(self, top_k = 5):
        prompts = clip.tokenize([f"a photo of car with {c} design" for c in self.keywords]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(prompts)
        text_features /= text_features.norm(dim = -1, keepdim = True)

        results = {}
        for image in tqdm(self.images):
            labels = self.get_labels(image)
    
            image_input = self.preprocess(Image.open(image).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim = -1, keepdim = True)

            similarity = (100.0 * image_features @ text_features.T)
            values, indices = similarity[0].topk(len(self.keywords))

            clsf_results = {}
            for value, index in zip(values, indices):
                clsf_results[self.keywords[index]] = value.item()
            
            results[image] = [clsf_results, labels]

        self.eval_metrics(results)

        with open("./results/ViT_L14_Results_Sampled.pkl", 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    model_name = "ViT-L14_Car"
    model_path = f"./checkpoints/{model_name}/interpolated/best_model_alpha_0.8_interpolated.pt"
    image_list = "./image_list/partial_test_images.pkl"
    inference = Inference(image_list, clip_model = "ViT-L/14", checkpoint = model_path)
    inference.run_inference()

    # with open("./results/RN101_Results.pkl", "rb") as f:
    #     results = pickle.load(f)
    # inference.eval_metrics(results)