import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sfm_reconstruct(image_paths, K): #K is your camera matrix
    """
    Performs Structure from Motion (SfM) reconstruction from a list of image paths.

    Args:
        image_paths: List of paths to the images.
        K: Camera intrinsic matrix (3x3 numpy array).

    Returns:
        points_3d: Reconstructed 3D points (Nx3 numpy array).
        camera_poses: Estimated camera poses (list of 4x4 transformation matrices).
       
    """
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()  # Brute-force matcher

    keypoints = []
    descriptors = []
    images = []

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Error loading image: {img_path}")
        images.append(img)
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)


    matches = []
    for i in range(len(images) - 1):
        m = bf.knnMatch(descriptors[i], descriptors[i+1], k=2) # k=2 for ratio test
        good_matches = []
        for m,n in m:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        matches.append(good_matches)




    camera_poses = []
    points_3d = []

    # Initialize first camera pose (world origin)
    camera_poses.append(np.eye(4))

    for i in range(len(matches)):
        pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints[i+1][m.trainIdx].pt for m in matches[i]]).reshape(-1, 1, 2)

        # Estimate Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)


        # Recover camera pose 
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)


        # Get next camera pose
        next_pose = np.hstack((R, t))
        if len(camera_poses)>0:
            next_pose = camera_poses[-1] @ next_pose  # Apply transformation relative to previous
        camera_poses.append(next_pose)



        # Triangulate points
        if i == 0: # Triangulate only from first pair for now (can extend for more views)
            proj_matrix1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            proj_matrix2 = K @ camera_poses[i+1][:3]  # Use estimated pose
            points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, pts1, pts2)
            points_3d_current = points_4d[:3] / points_4d[3]  # Homogeneous to Cartesian
            points_3d.append(points_3d_current)

    points_3d = np.hstack(points_3d)


    #Bundle adjustment would go here but is omitted for simplicity


    return points_3d, camera_poses





# Example Usage:
image_paths = ["i1.jpg", "i2.jpg", "i3.jpg"] # Add more image paths

# Example Camera matrix (replace with your calibrated K matrix)
K = np.array([[1000, 0, 500], [0, 1000, 300], [0, 0, 1]])


points_3d, camera_poses = sfm_reconstruct(image_paths, K)


# Visualize 3D points (using Open3D – you can use Matplotlib or other tools)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d.T)
o3d.visualization.draw_geometries([pcd])


# Print camera poses (optional)
for i, pose in enumerate(camera_poses):
    print(f"Camera Pose {i+1}:\n{pose}")