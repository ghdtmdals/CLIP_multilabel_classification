import clip
import torch
from torch import nn

class CLIPNetwork(nn.Module):
    def __init__(self, clip_model):
        super(CLIPNetwork, self).__init__()
        self.clip_model = clip_model
        self.model = self.load_model()
        
    def load_model(self):
        model, _ = clip.load(self.clip_model)
        model.float() # CLIP uses float16, convert to float
        ### Continue From
        # model.load_state_dict(torch.load('./checkpoints/ViT-B16_Car/best_model.pt')["model_state_dict"])

        return model
    
    ### 패딩에 해당하는 라벨은 학습할 필요가 없기 때문에 위치를 저장했다가 embedding 변환 결과를 0으로 치환해줌
    def get_padding_flags(self, texts):
        texts_sum = texts.sum(dim = -1)
        paddings = texts_sum == 0

        return paddings
    
    def forward(self, images, texts):
        padding_flags = self.get_padding_flags(texts)

        batch_size = texts.size(0)
        seq_len = texts.size(1)
        texts = texts.reshape(batch_size * seq_len, texts.size(2))

        text_features = self.model.encode_text(texts)
        text_features = text_features.reshape(batch_size, seq_len, text_features.size(1))
        text_features[padding_flags] = text_features[padding_flags] * 0 ### 패딩에 해당하는 embedding은 0으로 치환
        text_features = text_features.sum(dim = 1) ### Check Dims

        image_features = self.model.encode_image(images)

        image_features = image_features / image_features.norm(dim = 1, keepdim = True)
        text_features = text_features / text_features.norm(dim = 1, keepdim = True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text