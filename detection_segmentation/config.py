import os
from os.path import join, dirname, abspath


SCENE_NAMES = [
    'office_0',
    #'office_1',
    #'office_2',
    #'office_3',
    #'office_4',
    #'room_0',
    #'room_1',
    #'room_2',
]

TRAJECTORY_NAMES = [
    '00',
    # '01',
]

NUM_CLASSES = {
    'semantic_class': 256,
    'semantic_instance': 512,
}

NUM_FRAMES = 2000
SUBSAMPLE_STEP_SIZE = 4
