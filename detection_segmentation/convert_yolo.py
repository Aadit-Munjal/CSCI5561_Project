from ultralytics.data.converter import convert_coco
import os


folder = os.path.dirname(__file__)

file = os.path.join(folder, 'dataset/')
convert_coco(os.path.join(file, 'coco/'), 
             save_dir=os.path.join(file, 'dataset/'), 
             use_segments=True, use_keypoints=False, cls91to80=False)