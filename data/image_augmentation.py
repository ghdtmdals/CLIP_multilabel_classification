import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

import cv2
from PIL import Image

from enum import Enum
from tqdm import tqdm
from utils.utils import make_dir, read_files
import argparse



############### Script for Image Augmentation ###############
#######     Features: 
#######     1. LEFT-RIGHT Flip
#######     2. TOP-BOTTOM Flip
#######     3. Affine Transformation
#######     4. Sketch Conversion

#######     Sample Command Lines:
#######     Directory: ~/clip_finetune_main
#######         
#######     LEFT-to-Right Flip:
#######     python ./data/image_augmentation.py --image_path ./augmentation_samples/sample_cropped --save_path ./augmentation_samples --options 0
#######         
#######     Run All Augmentation Options:
#######     python ./data/image_augmentation.py --image_path ./augmentation_samples/sample_cropped --save_path ./augmentation_samples --options 4



class AugOptions(Enum):
    LEFT_RIGHT_FLIP = 0
    TOP_BOTTOM_FLIP = 1
    AFFINE = 2
    TO_SKETCH = 3
    ALL = 4

class AugmentImages:
    def __init__(self):
        self.args = self.get_args()
        self.options = AugOptions
        self.images = self.load_images()

        ### Affine Transformation Needs Tensor Type
        self.image_to_tensor = transforms.ToTensor()
        self.image_affine = transforms.RandomAffine(90, shear = self.args.shear)

        self.augment_functions = [self.left_right_flip, self.top_bottom_flip, self.affine, self.convert_to_sketch]
    
    def __call__(self):
        self.run_augmentation()

    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--image_path', required = True, type = str)
        parser.add_argument('--save_path', required = True, type = str)
        parser.add_argument('--options', required = True, type = int)
        parser.add_argument('--shear', default = 20) ### For Affine Transformation Only
        args = parser.parse_args()

        return args

    def load_images(self):
        #### Directly Reading Image Files From Folders, Folders Should Only Contain Image Files
        all_images = read_files(self.args.image_path)

        print("Total %d Images Loaded from %s" % (len(all_images), self.args.image_path))

        return all_images
    
    def run_augmentation(self):
        for image in tqdm(self.images):
            image_name = image.split('/')[-1]

            #### Run Augmentation Based on Selected Option
            selected_option = self.options(self.args.options)

            if selected_option is not self.options.ALL:
                self.augment_functions[selected_option.value](image)
            else:
                for aug_function in self.augment_functions:
                    aug_function(image)

    def left_right_flip(self, image_path: str):
        image_name = image_path.split('/')[-1]
        image = Image.open(image_path)
        hor_flip = image.transpose(Image.FLIP_LEFT_RIGHT)

        save_dir = "%s/hflip" % self.args.save_path
        make_dir(save_dir)

        save_name = "%s/hflip_%s" % (save_dir, image_name)
        hor_flip.save(save_name)

    def top_bottom_flip(self, image_path: str):
        image_name = image_path.split('/')[-1]
        image = Image.open(image_path)
        ver_flip = image.transpose(Image.FLIP_TOP_BOTTOM)

        save_dir = "%s/vflip" % self.args.save_path
        make_dir(save_dir)

        save_name = "%s/vflip_%s" % (save_dir, image_name)
        ver_flip.save(save_name)

    def affine(self, image_path: str):
        image_name = image_path.split('/')[-1]
        image = Image.open(image_path)
        tensor_transformed = self.image_to_tensor(image)
        affine_transformed = self.image_affine(tensor_transformed)
        affine_image = to_pil_image(affine_transformed)

        save_dir = "%s/affine" % self.args.save_path
        make_dir(save_dir)
        
        save_name = "%s/affine_%s" % (save_dir, image_name)
        affine_image.save(save_name)

    def convert_to_sketch(self, image_path: str):
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        gery_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        invert = cv2.bitwise_not(gery_img)
        blur = cv2.GaussianBlur(invert, (21, 21), 0)
        invertedblur = cv2.bitwise_not(blur)
        sketch = cv2.divide(gery_img, invertedblur, scale = 256.0)

        save_dir = "%s/sketch" % self.args.save_path
        make_dir(save_dir)

        save_name = "%s/sketch_%s" % (save_dir, image_name)
        cv2.imwrite(save_name, sketch)

if __name__ == "__main__":
    augmentation = AugmentImages()
    augmentation() #### Run on Call