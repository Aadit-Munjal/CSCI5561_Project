import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

from transformation_and_pose import get_correspondences, get_pose_with_pnp
from icp_scratch import icp

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp])

# Assuming we already have the intrinsic matrix

K = np.array([
        [600, 0, 600],
        [0, 600, 340],
        [0, 0, 1],
    ])

intrinsics = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600, 600, 600, 340)

cv2_color_images = []
cv2_depth_images = []

o3d_semantic_images = []
o3d_depth_images = []

for file in os.listdir("rgb"):
    cv2_color_images.append(cv2.imread(os.path.join('rgb', file)))
    cv2_depth_images.append(cv2.imread(os.path.join('depth',file), cv2.IMREAD_UNCHANGED))
    o3d_semantic_images.append(o3d.io.read_image(os.path.join('semantic_instance-viz',file)))
    o3d_depth_images.append(o3d.io.read_image(os.path.join('depth',file)))


extrinsics = []

chain_toggle = 0

for i in range(len(cv2_color_images)):
    if i == 0:
        extrinsics.append(np.identity(4))
    else:
        x1, x2 = get_correspondences(cv2_color_images[0], cv2_color_images[i], 0.5)
        
        if (np.shape(x1)[0] < 15):
            chain_toggle = 1
            
        if chain_toggle == 1:
            x1, x2 = get_correspondences(cv2_color_images[i-1], cv2_color_images[i], 0.5)
            extrinsics.append(get_pose_with_pnp(x1, x2, cv2_depth_images[i-1], K) @ extrinsics[i-1])
        else:
            extrinsics.append(get_pose_with_pnp(x1, x2, cv2_depth_images[0], K))



final_pcds = []

pre_icp_rmses = []
post_icp_rmses = []

pre_icp_fitnesses = []
post_icp_fitnesses = []

num_nonzero = 0

for image_num in range(len(extrinsics)):
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_semantic_images[image_num], o3d_depth_images[image_num],
                    convert_rgb_to_intensity=False
                    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image,
                        intrinsics,
                        extrinsic = extrinsics[image_num]
                        )
                    
    pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

    if image_num == 0:
        final_pcds.append(pcd)
    else:

        P = icp(pcd, final_pcds[0])
    
        evaluation = o3d.pipelines.registration.evaluate_registration(
            pcd, final_pcds[0], 0.02, np.identity(4))
        print("pre-icp")
        print(evaluation)
        
        if np.shape(evaluation.correspondence_set)[0] != 0:
            pre_icp_rmse = float(evaluation.inlier_rmse)
            pre_icp_rmses.append(pre_icp_rmse)
            pre_icp_fitnesses.append(evaluation.fitness)
            num_nonzero += 1
    

        
        evaluation = o3d.pipelines.registration.evaluate_registration(
            pcd, final_pcds[0], 0.02, P)
        print("post-icp")
        print(evaluation)

        if np.shape(evaluation.correspondence_set)[0] != 0:
            
            post_icp_rmse = float(evaluation.inlier_rmse)
            post_icp_rmses.append(post_icp_rmse)
            post_icp_fitnesses.append(evaluation.fitness)
        
        pcd.transform(P)
        final_pcds.append(pcd)


print(f"Average RMSE pre-ICP: {np.sum(np.array(pre_icp_rmses))/num_nonzero}")
print(f"Average RMSE post-ICP: {np.sum(np.array(post_icp_rmses))/num_nonzero}")

print(f"Average fitness pre-ICP: {np.sum(np.array(pre_icp_fitnesses))/num_nonzero}")
print(f"Average fitness post-ICP: {np.sum(np.array(post_icp_fitnesses))/num_nonzero}")


for i in range(len(final_pcds)):
    final_pcds[i] = final_pcds[i].voxel_down_sample(voxel_size=0.02)    
    
o3d.visualization.draw_geometries(final_pcds)

