import os
import json
import random
import shutil
from tqdm import tqdm
import requests
from pathlib import Path
import zipfile

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def extract_zip(zip_path, extract_path):
    """Extract zip file with progress bar"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total size for progress bar
        total_size = sum(info.file_size for info in zip_ref.filelist)
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        for file in zip_ref.filelist:
            zip_ref.extract(file, extract_path)
            progress_bar.update(file.file_size)
        progress_bar.close()

def create_coco_mini():
    """Create COCO2017-mini dataset"""
    try:
        # Create directories
        os.makedirs('data/coco', exist_ok=True)
        os.makedirs('data/coco/annotations', exist_ok=True)
        os.makedirs('data/coco/train2017', exist_ok=True)
        os.makedirs('data/coco/val2017', exist_ok=True)

        # Download COCO2017 annotations
        print("Downloading COCO2017 annotations...")
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        download_file(ann_url, "data/coco/annotations_trainval2017.zip")

        # Download COCO2017 images
        print("Downloading COCO2017 images...")
        train_url = "http://images.cocodataset.org/zips/train2017.zip"
        val_url = "http://images.cocodataset.org/zips/val2017.zip"
        
        download_file(train_url, "data/coco/train2017.zip")
        download_file(val_url, "data/coco/val2017.zip")

        # Extract files
        print("Extracting annotations...")
        extract_zip("data/coco/annotations_trainval2017.zip", "data/coco")
        print("Extracting training images...")
        extract_zip("data/coco/train2017.zip", "data/coco")
        print("Extracting validation images...")
        extract_zip("data/coco/val2017.zip", "data/coco")

        # Create mini dataset
        print("Creating COCO2017-mini...")
        
        # Load annotations
        with open('data/coco/annotations/instances_train2017.json', 'r') as f:
            train_ann = json.load(f)
        with open('data/coco/annotations/instances_val2017.json', 'r') as f:
            val_ann = json.load(f)

        # Select random images
        train_images = random.sample(train_ann['images'], 5000)
        val_images = random.sample(val_ann['images'], 500)

        # Get image IDs
        train_ids = {img['id'] for img in train_images}
        val_ids = {img['id'] for img in val_images}

        # Filter annotations
        train_ann['images'] = train_images
        train_ann['annotations'] = [ann for ann in train_ann['annotations'] 
                                  if ann['image_id'] in train_ids]
        
        val_ann['images'] = val_images
        val_ann['annotations'] = [ann for ann in val_ann['annotations'] 
                                if ann['image_id'] in val_ids]

        # Save mini annotations
        with open('data/coco/annotations/instances_train2017_mini.json', 'w') as f:
            json.dump(train_ann, f)
        with open('data/coco/annotations/instances_val2017_mini.json', 'w') as f:
            json.dump(val_ann, f)

        # Create mini image directories
        os.makedirs('data/coco/train2017_mini', exist_ok=True)
        os.makedirs('data/coco/val2017_mini', exist_ok=True)

        # Copy selected images
        print("Copying selected images...")
        for img in tqdm(train_images):
            src = f"data/coco/train2017/{img['file_name']}"
            dst = f"data/coco/train2017_mini/{img['file_name']}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        for img in tqdm(val_images):
            src = f"data/coco/val2017/{img['file_name']}"
            dst = f"data/coco/val2017_mini/{img['file_name']}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        # Clean up zip files
        os.remove("data/coco/annotations_trainval2017.zip")
        os.remove("data/coco/train2017.zip")
        os.remove("data/coco/val2017.zip")

        print("COCO2017-mini dataset created successfully!")
        
        # Verify the dataset
        train_dir = "data/coco/train2017_mini"
        val_dir = "data/coco/val2017_mini"
        train_images = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
        val_images = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
        print(f"Found {train_images} training images and {val_images} validation images")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    create_coco_mini() 