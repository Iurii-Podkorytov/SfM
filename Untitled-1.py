import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def process_pair(im1, im2, K):
    detector = cv2.SIFT_create()
    # detector = cv2.ORB_create()
    kp1, des1 = detector.detectAndCompute(im1, None)
    kp2, des2 = detector.detectAndCompute(im2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:  # Experiment with this threshold
            good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    try:
        fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 4, 0.99)
        essential_matrix = K.T @ fundamental_matrix @ K
        _, R, t, _ = cv2.recoverPose(essential_matrix, pts1, pts2, K)
        points3D = cv2.triangulatePoints(np.hstack((np.eye(3), np.zeros((3, 1)))), np.hstack((R, t)), pts1, pts2)
        points3D = cv2.convertPointsFromHomogeneous(points3D.T)[:,0,:]
        return points3D
    except Exception as e:
        print(f"Error processing image pair: {e}")
        return None

def sfm(images, K):
    all_points_3d = []
    for i in range(len(images) - 1):
        points_3d = process_pair(images[i], images[i + 1], K)
        if points_3d is not None:
            all_points_3d.append(points_3d)
        
    all_points_3d = np.vstack(all_points_3d)
    return all_points_3d


def plot_3d_points(points3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Triangulated 3D Points')
    plt.show()


images = []
folder_path = "frames"
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

K = np.array([[1000, 0, 500], [0, 1000, 300], [0, 0, 1]])  #REPLACE WITH YOUR ACTUAL K MATRIX
all_points_3d = sfm(images, K)
plot_3d_points(all_points_3d)