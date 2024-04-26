-- Project Architecture
    -- data
        -- coco_dataset.py: MS-COCO Dataset Class
        -- dataset.py: Car Image Dataset Class
        -- image_augmentation.py: Image Augmentation Script

    -- network
        -- clip_network.py: Implementation of Multi-label Classifition Version of CLIP

    -- scripts
        -- coco_inference.py: MS-COCO Dataset Multi-label Inference Script
        -- finetune.py: Finetuning Script for MS-COCO and Car Image Datasets
        -- inference.py: Car Image Dataset Multi-label Inference Script
        -- single_label_inference.py: MS-COCO Dataset Single-label Inference Script; For Test Purpose
    
    -- utils
        -- scheduler.py: Optimizer's Learning Rate Scheduler
        -- unnormalize.py: For Visualization of Normalized Image
        -- utils.py: Contains Utility Functions