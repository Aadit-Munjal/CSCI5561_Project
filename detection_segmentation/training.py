from ultralytics import YOLO
import os
import comet_ml
from comet_ml import Experiment
comet_ml.login(project_name="5561project",api_key="fill_in_your_api_key")

if __name__ == '__main__':
    model = YOLO("yolo11m-seg.pt")  # load a pretrained model (recommended for training)
    root = os.path.dirname(__file__)
    folder = os.path.join(root, 'dataset/dataset')
    file = os.path.join(folder,'dataset.yaml')
    results = model.train(data=file, 
                          epochs=50,
                          project='5561project',
                          save_period=1)