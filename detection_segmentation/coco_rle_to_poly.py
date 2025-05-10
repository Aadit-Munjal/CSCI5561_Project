import numpy as np
import cv2
from pycocotools import mask as maskUtils
import json
import os
from tqdm import tqdm

def rle_to_polygon(rle):
    """
    Converts a COCO RLE mask to polygon format.
    
    Arguments:
    - rle: The RLE (Run-Length Encoded) mask.

    Returns:
    - List of polygons, where each polygon is a list of points.
    """

    binary_mask = maskUtils.decode(rle)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()  # Flatten contour points into a list
        if len(contour) >= 0:  # COCO format requires at least 3 points (6 values)
            polygons.append(contour)
    return polygons

def convert_coco_rle_to_polygon(coco_annotations):
    """
    Convert all RLE annotations in a COCO dataset to polygon format.
    
    Arguments:
    - coco_annotations: A list of COCO annotations (with RLE masks).
    
    Returns:
    - Updated annotations with polygon masks.
    """

    for ann in tqdm(coco_annotations):
        if isinstance(ann['segmentation'], dict):
            rle = ann['segmentation']
            # Convert RLE to polygons
            polygons = rle_to_polygon(rle)
            if polygons:
                ann['segmentation'] = polygons
            else:
                print(f"Warning: Could not convert mask for annotation id {ann['id']}")
    return coco_annotations



if __name__ == '__main__':
    folder = os.path.dirname(__file__)
    os.makedirs(os.path.join(folder, 'dataset/coco'))
    ann_file_train = os.path.join(folder, 'dataset/coco_rle_train.json')
    out_file_train = os.path.join(folder, 'dataset/coco/coco_poly_train.json')
    ann_file_val = os.path.join(folder, 'dataset/coco_rle_val.json')
    out_file_val = os.path.join(folder, 'dataset/coco/coco_poly_val.json')


    with open(ann_file_train, 'r') as f:
        coco_data = json.load(f)

    coco_data['annotations'] = convert_coco_rle_to_polygon(coco_data['annotations'])

    with open(out_file_train, 'w') as f:
        json.dump(coco_data, f)
    

    with open(ann_file_val, 'r') as f:
        coco_data = json.load(f)

    coco_data['annotations'] = convert_coco_rle_to_polygon(coco_data['annotations'])

    with open(out_file_val, 'w') as f:
        json.dump(coco_data, f)

    print("COCO RLE to polygon conversion complete.")