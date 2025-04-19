import open3d as o3d
import numpy as np
import sys


def icp(target, source, trans_init):

    # Downsample pointclouds
    downpcd = target.voxel_down_sample(voxel_size=0.2)
    downpcd2 = source.voxel_down_sample(voxel_size=0.2)

    # Get points from point clouds
    points1 = np.asarray(downpcd.points)
    points2 = np.asarray(downpcd2.points)

    # Get correspondence set
    x1, x2 = find_correspondence_set(points1, points2)



def find_correspondence_set(points1, points2):
    n = np.shape(points1)[0]
    m = np.shape(points2)[0]

    x1 = []
    x2 = []

    for i in range(n):
        min_distance = sys.maxsize
        min_idx = -1
        for j in range(m):
            distance = np.linalg.norm(points1[i] - points2[j])
            if distance < min_distance:
                min_idx = j
                min_distance = distance

        x1.append(points1[i])
        x2.append(points2[min_idx])


    return x1, x2
