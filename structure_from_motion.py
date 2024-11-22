import os
import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

class SfM:
    def __init__(self, detector, matcher, K):
        self.detector = detector
        self.matcher = matcher
        self.K = K
        self.images = []
        self.points_2d = []
        self.camera_rotations = [np.eye(3)]
        self.camera_translations = [np.zeros((3, 1))]
        self.points_3d = []


    def _get_features(self, im1, im2):
        kp1, des1 = self.detector.detectAndCompute(im1, None)
        kp2, des2 = self.detector.detectAndCompute(im2, None)
        return kp1, des1, kp2, des2
    
    def _get_matched_points(self, kp1, kp2, des1, des2):
        matches = self.matcher.match(des1, des2)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        return pts1, pts2

    def _nonlinear_triangulation(self, R1, t1, R2, t2, pts1, pts2):
        def project_points(X, R, t):
            projected_points = self.K @ (R @ X.T + t)
            projected_points = projected_points[:2,:] / projected_points[2,:]
            return projected_points.T
        def reprojection_error(X, R1, t1, R2, t2, pts1, pts2):
            X = X.reshape(-1, 3)
            projected_pts1 = project_points(X, R1, t1)
            projected_pts2 = project_points(X, R2, t2)
            error = np.concatenate((projected_pts1 - pts1.reshape(-1,2), projected_pts2 - pts2.reshape(-1,2)), axis=1).flatten()
            return error
        def triangulation(kp1, kp2, T_1, T_2):
            kp1_3D = np.ones((3, kp1.shape[0]))
            kp2_3D = np.ones((3, kp2.shape[0]))
            kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
            kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()
            X = cv2.triangulatePoints(T_1[:3], T_2[:3], kp1_3D[:2], kp2_3D[:2])
            X /= X[3]
            return X[:3]
        
        T1 = self._to_homogeneous_matrix(R1, t1)
        T2 = self._to_homogeneous_matrix(R2, t2)
        points3D_initial = triangulation(pts1, pts2, T1, T2)
        
        initial_guess = points3D_initial.flatten()
        result = least_squares(reprojection_error, initial_guess,
                            args=(R1, t1, R2, t2, pts1.reshape(-1,2), pts2.reshape(-1,2)),
                            method='lm', xtol=1e-8)

        optimized_points = result.x.reshape(-1, 3)
        return optimized_points

    def _load_images(self, folder_path):
        image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg")])
        self.images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
    
    def _to_homogeneous_matrix(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    def reconstruct(self, images_folder):
        self._load_images(images_folder)
        points3d = []
        for i in range(len(self.images)-1):
            # Step 1: Extract features with feature_detector
            kp1, des1, kp2, des2 = self._get_features(self.images[i], self.images[i+1])

            # Step 2: Match features with matcher
            pts1, pts2 = self._get_matched_points(kp1, kp2, des1, des2)

            # Step 3: Estimate fundamental matrix
            fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 4, 0.99)
            essential_matrix = self.K.T @ fundamental_matrix @ self.K

            # Step 4: Find valid camera poses
            _, R, t, _ = cv2.recoverPose(essential_matrix, pts1, pts2)

            # Store points and camera poses
            self.points_2d.append((pts1, pts2))
            self.camera_rotations.append(R @ self.camera_rotations[-1])
            self.camera_translations.append(R @ self.camera_translations[-1] + t)
            
            # Step 5: Triangulation
            points3d.append(self._nonlinear_triangulation(
                np.eye(3), np.zeros((3,1)), 
                self.camera_rotations[-1], self.camera_translations[-1],
                pts1, pts2
            ))

        points3d = np.vstack(points3d)
        return points3d

def plot_3d_points(points3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Triangulated 3D Points')
    plt.show(block=True)

K = np.array([[1.59877590e+03, 0, 5.07401915e+02], [0, 1.57986433e+03, 7.23899817e+02], [0, 0, 1]])

sfm = SfM(cv2.SIFT_create(), cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True), K)
points3d = sfm.reconstruct("images")
plot_3d_points(points3d)