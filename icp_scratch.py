import open3d as o3d
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors


def icp(source, target, max_iter=20, tol=1e-5):
    
    # # Downsample pointclouds - set voxel size to 0.015 for better results
    downpcd = source.voxel_down_sample(voxel_size=0.02)
    downpcd2 = target.voxel_down_sample(voxel_size=0.02)

    # Get points from point clouds
    points1 = np.asarray(downpcd.points)
    points2 = np.asarray(downpcd2.points)

    P_final = np.identity(4)
    prev_error = sys.maxsize

    for i in range(max_iter):
        # Get correspondence set
        x1, x2 = find_correspondence_set_NN(points1, points2)

        if (np.shape(x1)[0] < 10):
            break

        curr_error = compute_error(x1, x2)
        
        # Exit early if change in error is within tolerance
        if abs(prev_error - curr_error) < tol:
            break
        

        # Center data
        x1_center = np.mean(x1, axis=0)
        x1_centered = x1 - x1_center
        x2_center = np.mean(x2, axis=0)
        x2_centered = x2 - x2_center

        # Compute cross covariance matrix and extract rotation and translation
        W = x2_centered.T @ x1_centered
        

        try:
            U, _, VT = np.linalg.svd(W)
        except:
            break
        
        R = U @ VT

        # Check for reflection
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ VT


        t = x2_center - R @ (x1_center)

        points1 = transform_points(points1, R, t)
        
        P = np.vstack((np.hstack((R, np.reshape(t, (3,1)))), np.array([0, 0, 0, 1])))
        P_final = P @ P_final

        prev_error = curr_error
        
    return P_final


def transform_points(points1, R, t):
    return (R @ points1.T).T + t

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


    return np.array(x1), np.array(x2)

def find_correspondence_set_NN(points1, points2, dist_thr=0.5):

    corresp_x2_to_x1 = []

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points1)
    distances, indices = nbrs.kneighbors(points2)

    for i in range(np.shape(indices)[0]):
        if distances[i][0] < dist_thr * distances[i][1]:
            corresp_x2_to_x1.append([indices[i][0], i])

    corresp_x1_to_x2 = []

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points2)
    distances, indices = nbrs.kneighbors(points1)

    for i in range(np.shape(indices)[0]):
        if distances[i][0] < dist_thr * distances[i][1]:
            corresp_x1_to_x2.append([i, indices[i][0]])


    final_indices = []
    for match in corresp_x2_to_x1:
        if match in corresp_x1_to_x2:
            final_indices.append(match)

    x1 = []
    x2 = []

    for i in range(len(final_indices)):
        x1.append(points1[final_indices[i][0]])
        x2.append(points2[final_indices[i][1]])

    return np.array(x1), np.array(x2)
            

def compute_error(x1, x2):
    return np.mean(np.square(x1 - x2)) / np.shape(x1)[0]


