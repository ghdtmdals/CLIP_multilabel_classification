import torch
from scripts.finetune import Finetune

def run_finetune(epochs, train_datapath, test_datapath, learning_rate, batch_size, device):
    finetune = Finetune(epochs, learning_rate, batch_size, device)
    finetune.train_loop(train_datapath, test_datapath)

if __name__ == "__main__":
    epochs = 30
    learning_rate = 1e-8
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train_datapath = "./image_list/train_images.pkl"
    # test_datapath = "./image_list/test_images.pkl"

    train_datapath = "./image_list/partial_train_images.pkl"
    test_datapath = "./image_list/partial_test_images.pkl"

    # train_datapath = "./image_list/train_images_no_sketch.pkl"
    # test_datapath = "./image_list/test_images_no_sketch.pkl"
    
    # train_datapath = "./image_list/coco_train_images.pkl"
    # test_datapath = "./image_list/coco_test_images.pkl"

    run_finetune(epochs, train_datapath, test_datapath, learning_rate, batch_size, device)