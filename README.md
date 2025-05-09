# Semantic 3D Reconstruction for Indoor AI Agents
By Aadit Munjal, Runqi Chen, Vivek Kumar, Ankit Kumar


## Goal and Approach

The primary goal is to implement a full 3D semantic reconstruction pipeline:

1.  **Key Frame Selection**: The dataset is subsampled to obtain a feasible number of images.
2.  **Instance Segmentation**: The 2D RGB images are instance-segmented by finetuning a pre-trained model.
3.  **Estimation of Extrinsic Parameters**:  The extrinsic parameters are obtained for each image pair using the Perspective-n-Point algorithm.
4.  **Point Cloud Generation**: Point clouds are generated based on the semantically segmented images and corresponding depth data.
5.  **ICP**:  The Iterative Closest Point algorithm is used to augment the alignment of the point clouds.
6.  **Visualization**: A unified point cloud consisting of aligned, down-sampled point clouds from all frames is visualized.

## Dataset

We utilized office 0 of the vMAP dataset processed from the Replica Dataset.

## Project Structure

```
CSCI5561_Project/
├── depth/                         # Depth data for 3D reconstruction (post keyframe selection)
├── detection_segmentation/        # Implementations of object detection, semantic segmentation, and instance segmentation
    ├── dataset/                   # Dataset file for finetuning instance segmentation model
    ├── demo/                      # Demonstration of COCO instance segmentation visualization
    ├── scratch/                   # Implementation of object detection and semantic segmentation from scratch in PyTorch
    ├── coco_rle_to_poly.py        # Convert COCO RLE label to polygon label
    ├── config.py                  # Dataset configuration file
    ├── convert_yolo.py            # Convert COCO polygon label to YOLO label
    ├── label_information.py       # Preprocess vMAP dataset label information
    ├── mask_to_coco_rle.py        # Convert grayscale segmentation masks to COCO RLE label
    ├── shift_index.py             # Decrement COCO semantic index to YOLO
    ├── training.py                # YOLO11 model training
    ├── validation.py              # YOLO11 model validation             
    └── visualize_coco.py          # Visualize COCO instance segmentation labels
├── rgb/                           # RGB data for 3D reconstruction (post keyframe selection)
├── semantic_instance-viz          # Ground truth segmentation data for 3D reconstruction (post keyframe selection)
├── README.md                      # This file
├── environment.yml                # Instance segmentation dependencies
├── icp_scratch.py                 # Implementation of point-to-point ICP from scratch
├── main.py                        # Main file for 3D reconstruction
├── requirements.txt               # 3D reconstruction dependencies
└── transformation_and_pose.py     # Code for extrinsics calibration
```


## Setup & Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Aadit-Munjal/CSCI5561_Project
    cd CSCI5561_Project
    ```
2.  Create and activate virtual environment (highly recommended):
   
    Open3D only supports python versions 3.8 - 3.12. The 3D reconstruction pipeline was tested and run on a windows machine with python 3.9.13 installed. The instance segmentation pipeline was tested and run on a windows machine with python 3.13.2 in conda environment. The two pipelines have conflicting dependency requirements and have to be run separately.
4.  To install all necessary packages for 3D reconstruction, the following command can be utilized:
    ```bash
    pip install -r requirements.txt
    ```
    To install all necessary packages for instance segmentation, the following command can be utilized:
    ```bash
    conda env update -f environment.yml
    ```




## 3D Reconstruction

To run the 3D reconstruction component of the project use the following command:
```bash
python3 main.py
```
