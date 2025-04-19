import cv2
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt


# Assuming we already have the intrinsic matrix

K = np.array([
        [600, 0, 600],
        [0, 600, 340],
        [0, 0, 1],
    ])

# Get correspondences using SIFT and nearest neighbor
def get_correspondences(img1, img2, dist_thr):
    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    corresp_x2_to_x1 = []

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(descriptors1)
    distances, indices = nbrs.kneighbors(descriptors2)

    for i in range(np.shape(indices)[0]):
        if distances[i][0] < dist_thr * distances[i][1]:
            corresp_x2_to_x1.append([indices[i][0], i])

    corresp_x1_to_x2 = []

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(descriptors2)
    distances, indices = nbrs.kneighbors(descriptors1)

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
        x1.append(keypoints1[final_indices[i][0]])
        x2.append(keypoints2[final_indices[i][1]])

    x1 = cv2.KeyPoint_convert(np.array(x1))
    x2 = cv2.KeyPoint_convert(np.array(x2))

    return x1, x2

# Returns the transformation that gets 3D points in camera 2's coordinate frame to camera 1 's coordinate frame
def get_transformation(points1, points2):
    F, _ = cv2.findFundamentalMat(points1, points2, method=cv2.FM_RANSAC)

    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])


    pts3Ds = []
    P1 = K @ (np.hstack((np.identity(3), np.ones((3,1)))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -1*Rs[i] @ Cs[i]))
        pts3D = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        pts3D = pts3D[:3, :]/pts3D[3,:]
        pts3D = pts3D.T
        pts3Ds.append(pts3D)


    best_num_inliers = 0
    best_R = np.zeros((3,3))
    best_C = np.zeros((3,1))

    for i in range(len(Rs)):
        R = Rs[i]
        C = Cs[i]

        pts3D = pts3Ds[i]
        
        num_inliers = 0
        for j in range(np.shape(pts3D)[0]):
            x1 = np.resize(pts3D[j], (3,1))
            x2 = R @ ((np.resize(pts3D[j], (3,1))) - C)

            if float(x1[2][0]) > 0 and float(x2[2][0]) > 0:
                num_inliers += 1

        if num_inliers > best_num_inliers:
            best_R = R
            best_C = C
            best_num_inliers = num_inliers


    return np.hstack((best_R, -1*best_R @ best_C))



def visualize_find_match(img1, img2, x1, x2, img_h=500, filename=None):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h / img1.shape[0]
    scale_factor2 = img_h / img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    if filename:   
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def proj_to_3d(pts2d, depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pts3d = []
    for point in pts2d:
        u,v = point
        z = depth[int(v), int(u)]/1000
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts3d.append([x, y, z])
    return np.array(pts3d)


def get_pose_with_pnp(points1, points2, depth1, K):
    
    pts3d_1 = proj_to_3d(points1, depth1, K)

    _, rvec, tvec, _ = cv2.solvePnPRansac(pts3d_1, points2, K, distCoeffs=None)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    
    P1 = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))

    return P1










