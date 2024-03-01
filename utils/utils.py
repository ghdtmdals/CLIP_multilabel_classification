import os
import random
import pickle
import math

import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

from utils.unnormalize import UnNormalize


################# Try Exception Control #################
# Executes commands and handles exceptions if any occurs
# Usage:
# with ignoring(Exception):
#     function()
@contextmanager
def ignoring(*exceptions):
    try:
        yield
    except exceptions as e:
        print(e)

################# Single Line Print for Looping #################
# Prints contexts on a single line for iterated process
# Usage:
# rprint("print something %d" % (100))
def rprint(context: str):
    print('\r{}'.format(context), end = "")

################# Fixed Seeds #################
# Fix random seeds in pytorch for reproducibility
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

################# Reads Files Under Root #################
# Reads all images under root path
def read_files(root: str):
    # all_files = []
    # for path, subdirs, files in tqdm(os.walk(root)):
    #     for i, name in enumerate(files):
    #         all_files.append(os.path.join(path, name))
    #         # if i > 1500:
    #         #     break
    all_files = os.listdir(root)
    
    all_paths = []
    for path in all_files:
        all_paths.append('%s/%s' % (root, path))

    return all_paths

################# Creates Directories #################
# Creates all paths if not present
def make_dir(path: str):
    if os.path.isdir(path) == False:
        os.makedirs(path)

################# Multiprocessing Function #################
# Multiprocessing executor for list type data processing
# Usage:
# def some_function(start, end):
#     for element in global_list[start:end]:
#         do something

# if __name__ == "__main__" :
#     multi_function(some_function, global_list, use_ratio)
def multi_function(exec_function, data: list, use_ratio: float = 0.5):
    n_cpu = int(multiprocessing.cpu_count() * use_ratio)
    full_len = len(data) # data count
    process_index = int(full_len / n_cpu) # split counts
    rng_list = [(i + 1) * process_index for i in range(n_cpu)] # split indicies
    rng_list[-1] = full_len
    if rng_list[0] != 0:  # add 0 on first index
        rng_list.insert(0, 0)
    if rng_list[-1] < full_len: # last element of range list should equal to data length
        rng_list.append(full_len)
    
    with ProcessPoolExecutor(max_workers = n_cpu) as executor:
        list(executor.map(exec_function, rng_list[0:-1], rng_list[1:]))

################# Reverse Dictionary #################
# Reverse keys and values of a dictionary
def invert_dictionary(obj: dict):
    inv_obj = defaultdict(list)

    for key, value in obj.items():
        inv_obj[value].append(key)

    return dict(inv_obj)

################# Save Test Results #################
# Save subplots of images, labels and corresponding predictions
def save_figure(epoch, n_iter, image_batch, output_results, save_path):
    unnorm = UnNormalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    images = []
    text_results = []
    for t, image in enumerate(image_batch):
        result = output_results[image]
        result_str = ""
        for i, (k, v) in enumerate(result.items()):
            if i == 0:
                result_str += f"{k}: {100 * v:.2f}%"
            else:
                result_str += f"\n{k}: {100 * v:.2f}%"
            if i > 10:
                break
        car_image = Image.open(image).convert('RGB').resize((224, 224))

        images.append(car_image)
        text_results.append(result_str)
    
    plt.switch_backend('agg')
    rows, cols = int(math.sqrt(len(image_batch))), int(math.sqrt(len(image_batch)))
    if (rows * cols) < len(image_batch):
        rows += 1; cols += 1
    fig = plt.figure(figsize = (100, 100), constrained_layout = True)

    for i, (image, text_result) in enumerate(zip(images, text_results)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image)
        ax.text(230, 160, text_result, size = 60)
        ax.axis('off')

    image_type = save_path.split('/')[-1]
    save_name = '%s/%s_%d.png' % (save_path, image_type, n_iter)
    plt.savefig(save_name)
    plt.close()

