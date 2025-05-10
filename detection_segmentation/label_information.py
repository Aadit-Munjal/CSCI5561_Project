"""Generate instance segmentation label information from vMAP info_semantic.json"""


import json
import os
import config
from tqdm import tqdm, trange

from config import SCENE_NAMES

DATA_ROOT = os.path.dirname(__file__)
SOURCE_FILE_PATH = os.path.join(DATA_ROOT, 'dataset/{scene_name}')
DES_FILE_PATH = os.path.join(DATA_ROOT, 'dataset')

def main():
    class_dict = {}
    for scene_name in tqdm(SCENE_NAMES):
        fmt = {
                'scene_name': scene_name,
            }
        raw_dir = SOURCE_FILE_PATH.format(**fmt)
        info_file = os.path.join(raw_dir,'info_semantic.json')
        with open(info_file, 'r') as f:
            info = json.load(f)
        for cls in info['classes']:
            class_dict[cls['id']] = cls['name']
    des_dir = DES_FILE_PATH.format(**fmt)
    os.makedirs(des_dir, exist_ok=True)
    class_ids = sorted(list(class_dict.keys()))
    sorted_class_dict = {i: class_dict[i] for i in class_ids}
    id_to_label = info['id_to_label']
    id_to_label_dict = {'id_to_label':id_to_label}
    class_file = os.path.join(des_dir, 'semantic_classes.json')
    id_to_label_file = os.path.join(des_dir, 'id_to_label.json')
    with open(class_file, 'w') as f:
        json.dump(sorted_class_dict, f, indent=4)
    with open(id_to_label_file, 'w') as f:
        json.dump(id_to_label_dict, f, indent=4)
    


if __name__ == '__main__':
    main()