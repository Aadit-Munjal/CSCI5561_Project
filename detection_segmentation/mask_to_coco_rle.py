"""Convert binary masks to COCO RLE instance segmentation format."""
"""Assume images are named as rgb_1.png, rgb_2.png, ..."""


import os
import json
import numpy as np
import datetime
from pycocotools import mask as maskUtils
from PIL import Image
import glob
from tqdm import tqdm



def create_coco_dataset(masks_dir, categories, output_json_path, id_to_label):
    """
    Convert a directory of segmentation masks to COCO format
    
    Args:
        masks_dir: Directory containing mask images
        categories: List of dictionaries with category info (id, name, supercategory)
        output_json_path: Path to save the COCO JSON file
        id_to_label: List mapping object id to class label
    """
    # Initialize COCO format structure
    coco_output = {
        "info": {
            "description": "Dataset in COCO format converted from segmentation masks",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    # Get all masks
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    
    ann_id = 1  # Start annotation IDs at 1

    # Process each mask file
    for img_id, mask_file in enumerate(tqdm(mask_files), 1):
        filename = os.path.basename(mask_file)
        
        # You can modify this to match your image naming convention
        image_filename = filename
        num = filename.split('_')[-1]
        image_filename = f"rgb_{num}"

        # Get image dimensions from the mask
        mask_img = Image.open(mask_file)
        width, height = mask_img.size

        # Add image info
        coco_output["images"].append({
            "id": img_id,
            "license": 1,
            "file_name": image_filename,
            "height": height,
            "width": width,
            "date_captured": ""
        })

        # Process each category in the mask
        mask_data = np.array(mask_img)

        for object_id in range(len(id_to_label)):
            binary_mask = (mask_data == object_id).astype(np.uint8)
        
            if np.sum(binary_mask) == 0:
                continue

            rle = maskUtils.encode(np.asfortranarray(binary_mask))
            rle['counts'] = rle['counts'].decode('utf-8')

            area = int(maskUtils.area(rle))
            bounds = maskUtils.toBbox(rle).tolist()

            category = id_to_label[object_id]
            if category <= 0:
                continue

            annotation = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category,
                    "segmentation": rle,
                    "area": area,
                    "bbox": bounds,  # [x, y, width, height]
                    "iscrowd": 0
                }
        
            coco_output["annotations"].append(annotation)
            ann_id += 1

    with open(output_json_path, 'w') as f:
        json.dump(coco_output, f)
        
    print(f"Conversion complete! COCO JSON saved to {output_json_path}")
    print(f"Total images: {len(coco_output['images'])}")
    print(f"Total annotations: {len(coco_output['annotations'])}")

if __name__ == '__main__':

    folder = os.path.dirname(__file__)
    masks_folder_train = os.path.join(folder,"dataset/masks/train")
    masks_folder_val = os.path.join(folder,"dataset/masks/val")
    images_folder = os.path.join(folder, "dataset/images")
    JASON_FILE_NAME = os.path.join(folder, "dataset/semantic_classes.json")
    ID_FILE_NAME = os.path.join(folder, "dataset/id_to_label.json")
    output_path_train = os.path.join(folder, "dataset", "coco_rle_train.json")
    output_path_val = os.path.join(folder, "dataset", "coco_rle_val.json")
    
    with open(JASON_FILE_NAME, 'r') as f:
        semantic_classes = json.load(f)

    categories = [{"id": int(key), "name": value, "supercategory": value} for key, value in semantic_classes.items()]
    with open(ID_FILE_NAME, 'r') as f:
        info = json.load(f)
    id_to_label = info['id_to_label']

    create_coco_dataset(
        masks_dir=masks_folder_train,
        categories=categories,
        output_json_path=output_path_train,
        id_to_label = id_to_label
    )

    create_coco_dataset(
        masks_dir=masks_folder_val,
        categories=categories,
        output_json_path=output_path_val,
        id_to_label = id_to_label
    )
