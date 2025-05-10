"""Visualize COCO instance segmentation masks on top of original image."""

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import random
import matplotlib.patches as patches


def visualize_coco(box):
    folder = os.path.dirname(__file__)
    ann_file = os.path.join(folder, 'demo/vis_coco_poly.json')

    # Uncommet to test COCO Run Length Encoding format
    # ann_file = os.path.join(folder, 'dataset/coco_rle.json')

    img_dir = os.path.join(folder, 'demo')

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        masked_image = image.copy()
        for ann in anns:
            color = tuple(random.randint(0, 255) for _ in range(3))
            if isinstance(ann['segmentation'], dict):
                mask = maskUtils.decode(ann['segmentation'])
                
                
                for c in range(3):
                    masked_image[:,:,c] = np.where(mask==1, color[c], masked_image[:,:,c])
            elif isinstance(ann['segmentation'], list):
                mask = np.zeros((img_info['height'],img_info['width']), dtype=np.uint8)
                for seg in ann['segmentation']:
                    pts = np.array(seg).reshape((-1,2))
                    cv2.fillPoly(mask, [pts], 1)
                    for c in range(3):
                        masked_image[:,:,c] = np.where(mask==1, color[c], masked_image[:,:,c])
            
            if box:
                bbox = ann['bbox']
                
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2,
                    edgecolor=[c/255 for c in color],
                    facecolor='None'
                )
                ax = plt.gca()
                ax.add_patch(rect)
                cat_id = ann['category_id']
                cat_name = coco.loadCats(cat_id)[0]['name']
                ax.text(
                    bbox[0], bbox[1]-5,
                    f'{cat_name}',
                    color = [c/255 for c in color],
                )
        
               


        plt.axis('off')
        plt.imshow(masked_image)
        
        plt.savefig(os.path.join(img_dir,'coco_label.png'), bbox_inches='tight', pad_inches=0)
        plt.show()
        
visualize_coco(True)