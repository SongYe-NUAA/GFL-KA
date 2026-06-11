import os
import json
import random
import shutil
from tqdm import tqdm

def create_coco_mini():
    """Create COCO2017-mini dataset from existing COCO2017 dataset"""
    try:
        # Create mini directories
        os.makedirs('data/coco_mini/train2017', exist_ok=True)
        os.makedirs('data/coco_mini/val2017', exist_ok=True)
        os.makedirs('data/coco_mini/annotations', exist_ok=True)

        # Load annotations
        print("Loading annotations...")
        with open('data/coco/annotations/instances_train2017.json', 'r') as f:
            train_ann = json.load(f)
        with open('data/coco/annotations/instances_val2017.json', 'r') as f:
            val_ann = json.load(f)

        # Select random images
        print("Selecting random images...")
        train_images = random.sample(train_ann['images'], 5000)
        val_images = random.sample(val_ann['images'], 500)

        # Get image IDs
        train_ids = {img['id'] for img in train_images}
        val_ids = {img['id'] for img in val_images}

        # Filter annotations
        print("Filtering annotations...")
        train_ann['images'] = train_images
        train_ann['annotations'] = [ann for ann in train_ann['annotations'] 
                                  if ann['image_id'] in train_ids]
        
        val_ann['images'] = val_images
        val_ann['annotations'] = [ann for ann in val_ann['annotations'] 
                                if ann['image_id'] in val_ids]

        # Save mini annotations
        print("Saving mini annotations...")
        with open('data/coco_mini/annotations/instances_train2017.json', 'w') as f:
            json.dump(train_ann, f)
        with open('data/coco_mini/annotations/instances_val2017.json', 'w') as f:
            json.dump(val_ann, f)

        # Copy selected images
        print("Copying training images...")
        for img in tqdm(train_images):
            src = f"data/coco/train2017/{img['file_name']}"
            dst = f"data/coco_mini/train2017/{img['file_name']}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        print("Copying validation images...")
        for img in tqdm(val_images):
            src = f"data/coco/val2017/{img['file_name']}"
            dst = f"data/coco_mini/val2017/{img['file_name']}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        print("COCO2017-mini dataset created successfully!")
        
        # Verify the dataset
        train_dir = "data/coco_mini/train2017"
        val_dir = "data/coco_mini/val2017"
        train_images = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
        val_images = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
        print(f"Found {train_images} training images and {val_images} validation images")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    create_coco_mini() 