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
        
        # images = random.sample(images, 100)

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
    
    def eval_metrics(self, results, is_sbs = False):
        ### Precision, Recall, F1 Scores
        y_pred = np.zeros((len(results), len(self.keywords)))
        y_true = np.zeros((len(results), len(self.keywords)))

        all_preds_ids = []
        for i, (_, result) in enumerate(results.items()):
            temp_labels = result[1]
            if is_sbs:
                temp_preds = list(result[0].keys())
            else:
                temp_preds = self.top_remove_softmax(result[0], threshold = 0.6)

            for label in temp_labels:
                y_true[i, self.keywords.index(label)] = 1
            
            for pred in temp_preds:
                y_pred[i, self.keywords.index(pred)] = 1
                all_preds_ids.append(self.keywords.index(pred))

        print(classification_report(y_true, y_pred, target_names = self.keywords, labels = np.unique(all_preds_ids), digits = 4))
    
    def top_remove_softmax(self, predictions, threshold = 0.5):
        ### Softmax 수행 후 가장 확률이 높은 하나를 예측 결과로 선정
        ### 해당 결과를 제외하고 동일한 과정을 반복
        ### 가장 확률이 높은 결과가 Threshold 보다 낮으면 종료
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
    
    ### Softmax 결과 대신 Similarity를 이용해 Inference 수행
    ### Threshold 설정이 모델 Backbone 별로 조금씩 변해서 활용하지 않음
    def greedy_inference(self):
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
    
    ### Beam Search의 개념을 차용해 Inference 수행
    ### Similarity가 가장 높은 예측 결과의 Text Embedding을 다른 타겟 클래스들의 Text Embedding과 합한 뒤에 다시 Similarity를 구하는 과정 반복
    ### 예측 결과 선정 행태가 top_remove_softmax와 차이가 없기 때문에 과정이 더 단순한 top_remove_softmax 사용
    def single_beam_search(self, save_path):
        prompts = clip.tokenize([f"a photo of car with {c} design" for c in self.keywords]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(prompts)

        results = {}
        for image in tqdm(self.images):
            labels = self.get_labels(image)
            image_input = self.preprocess(Image.open(image).convert("RGB")).unsqueeze(0).to(self.device)
            temp_text_features = text_features.clone()

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim = -1, keepdim = True)

            text_normalized = temp_text_features / temp_text_features.norm(dim = -1, keepdim = True)

            similarity = (100.0 * image_features @ text_normalized.T)
            value, index = similarity[0].topk(1)

            indices = [index]
            values = [value]

            while True:
                ### Sum
                temp_text_features = temp_text_features + temp_text_features[index]
                ### Mask
                ### 예측 결과로 선정된 타겟 클래스에 아주 작은 값으로 Masking 수행
                ### 0으로 치환할 경우 Cosine Similarity 연산 과정에 0 Divison 문제가 발생함
                temp_text_features[index] = 1e-20

                ### Inference 과정에서 평균적으로 4개의 예측 결과가 선정되면 
                ### 이후 선정되는 타겟 클래스들이 앞서 Masking한 타겟 클래스인 현상이 발생
                ### Ex) 4개 선정 결과: [0, 1, 3, 2] -> 5번째 선정 결과: [0, 1, 3, 2, 1]
                ### 이는 나머지 타겟 클래스가 아주 작은 Constant로 구성된 벡터보다 이미지 벡터와의 유사도가 낮다는 것을 의미
                ### 따라서 Beam Search나 Threshold를 지정하는 방식과는 다르게
                ### 중복된 타겟 클래스가 선정되면 Inference 과정을 종료하도록 구성

                text_normalized = temp_text_features / temp_text_features.norm(dim = -1, keepdim = True)
                similarity = (100.0 * image_features @ text_normalized.T)
                value, index = similarity[0].topk(1)
                
                if index in indices:
                    break
                
                indices.append(index)
                values.append(value)
            
            clsf_results = {}
            for value, index in zip(values, indices):
                clsf_results[self.keywords[index]] = value.item()
            
            results[image] = [clsf_results, labels]
        
        self.eval_metrics(results, is_sbs = True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    model_name = "ViT-B32_Car"
    model_path = f"./checkpoints/{model_name}/interpolated/best_model_alpha_0.8_interpolated.pt"
    image_list = "./image_list/partial_test_images.pkl"

    inference = Inference(image_list, clip_model = "ViT-B/32", checkpoint = model_path)
    # inference.greedy_inference()
    inference.single_beam_search(save_path = "./results/ViT_B32_BSResults.pkl")

    # with open("./results/RN101_Results.pkl", "rb") as f:
    #     results = pickle.load(f)
    # inference.eval_metrics(results)