# This code has been adapted from the code provided in open3D's documentation

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import copy


color_raw = o3d.io.read_image("semantic_class_0.png")
depth_raw = o3d.io.read_image("depth_0.png")

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw,
    convert_rgb_to_intensity=False
)


plt.subplot(1, 2, 1)
plt.title("Color")
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title("Depth")
plt.imshow(rgbd_image.depth)
plt.show()

intrinsics = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600, 600, 600, 340)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    intrinsics
)

pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])


o3d.visualization.draw_geometries([pcd], window_name="img0 point cloud")


color_raw = o3d.io.read_image("semantic_class_1.png")
depth_raw = o3d.io.read_image("depth_1.png")

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw,
    convert_rgb_to_intensity=False
)


plt.subplot(1, 2, 1)
plt.title("Color")
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title("Depth")
plt.imshow(rgbd_image.depth)
plt.show()

intrinsics = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600, 600, 600, 340)

pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    intrinsics
)

pcd2.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])


o3d.visualization.draw_geometries([pcd2], window_name="img1 point cloud")



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp])

source = pcd
target = pcd2


threshold = 0.02
trans_init = np.asarray(np.identity(4))


draw_registration_result(source, target, trans_init)


print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
print(evaluation)


print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)




