import clip
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from data.dataset import ImageDataset
from data.coco_dataset import COCODataset

from network.clip_network import CLIPNetwork

from utils.utils import rprint
from utils.scheduler import CosineAnnealingWarmUpRestarts
import pickle
from tqdm import tqdm

class Finetune:
    def __init__(self, epochs, learning_rate, batch_size, device):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

    def load_data(self, data_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # dataset = COCODataset(data_path = data_path, transform = transform)
        dataset = ImageDataset(data_path, transform)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, num_workers = 6)

        return dataloader, len(dataset)

    def train_setup(self):
        model = CLIPNetwork(clip_model = "ViT-L/14").to(self.device)
        # 'RN101',
        # 'ViT-B/32',
        # 'ViT-B/16',
        # 'ViT-L/14',
        
        image_criterion = nn.CrossEntropyLoss()
        text_criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = self.learning_rate, betas = (0.9, 0.98), eps = 1e-6, weight_decay = 0.001)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0 = 100, T_mult = 1, eta_max = 1e-6,  T_up = 10, gamma = 0.5)

        return model, image_criterion, text_criterion, optimizer, scheduler

    def train_loop(self, train_datapath, test_datapath):
        train_dataloader, train_size = self.load_data(train_datapath)
        test_dataloader, test_size = self.load_data(test_datapath)
        print("Total Number of Images: %d | Train Dataset: %d | Test Dataset: %d" % (train_size + test_size, train_size, test_size))

        model, image_criterion, text_criterion, optimizer, scheduler = self.train_setup()

        model_save_path = "./checkpoints/ViT-L14_Car"

        n_iter = 0
        running_loss = 0
        running_img_loss = 0
        running_txt_loss = 0
        img_acc = 0
        txt_acc = 0
        best_test_loss = 1e5
        model.train()

        print("Start Training for %d Epochs" % self.epochs)
        for epoch in range(self.epochs):
            for i, (images, texts) in enumerate(train_dataloader):
                images = images.to(self.device)
                texts = texts.to(self.device)

                logits_per_image, logits_per_text = model(images, texts)

                ground_truth = torch.arange(images.size(0)).to(self.device)

                image_loss = image_criterion(logits_per_image, ground_truth)
                text_loss = text_criterion(logits_per_text, ground_truth)
                loss = (image_loss + text_loss) / 2

                img_acc += (sum(logits_per_image.argmax(dim = 1) == ground_truth) / len(ground_truth)).item()
                txt_acc += (sum(logits_per_text.argmax(dim = 1) == ground_truth) / len(ground_truth)).item()

                running_loss += loss.item()
                running_img_loss += image_loss.item()
                running_txt_loss += text_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n_iter += 1

                if i % 100 == 0:
                    avg_loss = running_loss / n_iter
                    avg_img_loss = running_img_loss / n_iter
                    avg_txt_loss = running_txt_loss / n_iter
                    
                    avg_img_acc = img_acc / n_iter
                    avg_txt_acc = txt_acc / n_iter
                    avg_acc = (avg_img_acc + avg_txt_acc) / 2

                    rprint("Epoch: %d|Iter: %d|Avg Loss: %.4f|Avg Img Loss: %.4f|Avg Txt Loss: %.4f|Avg Acc: %.4f|Avg Img Acc: %.4f|Avg Txt Acc: %.4f" \
                            % (epoch, i, avg_loss, avg_img_loss, avg_txt_loss, avg_acc, avg_img_acc, avg_txt_acc))
                    
                    running_loss = 0
                    running_img_loss = 0
                    running_txt_loss = 0
                    img_acc = 0
                    txt_acc = 0
                    n_iter = 0

            test_iter = 0
            test_loss = 0
            test_img_loss = 0
            test_txt_loss = 0
            test_img_acc = 0
            test_txt_acc = 0
            with torch.no_grad():
                model.eval()
                for i, (images, texts) in enumerate(test_dataloader):
                    images = images.to(self.device)
                    texts = texts.to(self.device)

                    logits_per_image, logits_per_text = model(images, texts)

                    ground_truth = torch.arange(images.size(0)).to(self.device)

                    image_loss = image_criterion(logits_per_image, ground_truth)
                    text_loss = text_criterion(logits_per_text, ground_truth)
                    loss = (image_loss + text_loss) / 2

                    test_img_acc += (sum(logits_per_image.argmax(dim = 1) == ground_truth) / len(ground_truth)).item()
                    test_txt_acc += (sum(logits_per_text.argmax(dim = 1) == ground_truth) / len(ground_truth)).item()

                    test_loss += loss.item()
                    test_img_loss += image_loss.item()
                    test_txt_loss += text_loss.item()

                    test_iter += 1

            avg_test_loss = test_loss / test_iter
            avg_test_img_loss = test_img_loss / test_iter
            avg_test_txt_loss = test_txt_loss / test_iter

            avg_test_img_acc = test_img_acc / test_iter
            avg_test_txt_acc = test_txt_acc / test_iter
            avg_test_acc = (avg_test_img_acc + avg_test_txt_acc) / 2

            print("\nAvg Test Loss: %.4f|Avg Img Loss: %.4f|Avg Txt Loss: %.4f|Avg Test Acc: %.4f|Avg Test Img Acc: %.4f|Avg Test Txt Acc: %.4f" \
                % (avg_test_loss, avg_test_img_loss, avg_test_txt_loss, avg_test_acc, avg_test_img_acc, avg_test_txt_acc))

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save({"best episode": epoch,
                            "avg_test_loss": avg_test_loss,
                            "model_state_dict": model.model.state_dict()},
                            "%s/best_model.pt" % model_save_path)

            torch.save({"episode": epoch,
                        "avg_test_loss": avg_test_loss,
                        "model_state_dict": model.model.state_dict()},
                        "%s/last_model.pt" % model_save_path)