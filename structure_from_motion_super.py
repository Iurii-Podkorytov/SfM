import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from wrappers import *
from utils import *

class SfM:
    def __init__(self, detector, matcher, K, dist):
        self.detector = detector
        self.matcher = matcher
        self.K = K
        self.dist = dist
        self.images = []
        self.points_2d = []
        self.camera_rotations = [np.eye(3)]
        self.camera_translations = [np.zeros((3, 1))]
        self.points_3d = []

    def _get_features(self, im1, im2):
        kp1, des1, scores1 = self.detector.detect(im1)
        kp2, des2, scores2 = self.detector.detect(im2)
        return kp1, des1, kp2, des2, scores1, scores2
    
    def _get_matched_points(self, kp1, kp2, des1, des2, scores1, scores2, im1, im2):
        pts1, pts2 = self.matcher.match(kp1, des1, kp2, des2, scores1, scores2, im1, im2)
        return np.float32(pts1), np.float32(pts2)

    def _project_points(self, X, R, t):
        projected_points = self.K @ (R @ X.T + t)
        projected_points = cv2.convertPointsFromHomogeneous(projected_points.T)[:,0,:]
        return projected_points
    
    def _reprojection_error(self, X, R1, t1, R2, t2, pts1, pts2):
        X = X.reshape(-1, 3)
        projected_pts1 = self._project_points(X, R1, t1)
        projected_pts2 = self._project_points(X, R2, t2)
        error = np.concatenate(((projected_pts1 - pts1), (projected_pts2 - pts2))).ravel()
        return error
    
    def _triangulate(self, pts1, pts2, R1, t1, R2, t2):
        img1ptsHom = cv2.convertPointsToHomogeneous(pts1)[:,0,:]
        img2ptsHom = cv2.convertPointsToHomogeneous(pts2)[:,0,:]

        img1ptsNorm = (np.linalg.inv(self.K).dot(img1ptsHom.T)).T
        img2ptsNorm = (np.linalg.inv(self.K).dot(img2ptsHom.T)).T

        img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
        img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

        pts4d = cv2.triangulatePoints(np.hstack((R1,t1)),np.hstack((R2,t2)),
                                        img1ptsNorm.T,img2ptsNorm.T)
        pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]
        return pts3d

    def nonlinear_triangulation(self, points_3d, R1, t1, R2, t2, pts1, pts2):
        X0 = points_3d.flatten()
        optimized_params = least_squares(
            self._reprojection_error, X0, args=(R1, t1, R2, t2, pts1, pts2),
            max_nfev=50, method='lm'
        )
        optimized_points_3d = optimized_params.x.reshape(-1, 3)
        return optimized_points_3d
    
    def _load_images(self, folder_path):
        image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")] )
        self.images = [cv2.imread(p) for p in image_paths]

    def reconstruct(self, images_folder):
        self._load_images(images_folder)
        for i in range(len(self.images)-1):
            # Detect fratures on two consecutive pictures and match them
            kp1, des1, kp2, des2, scores1, scores2 = self._get_features(self.images[i], self.images[i+1])
            pts1, pts2 = self._get_matched_points(kp1, kp2, des1, des2, scores1, scores2, self.images[i], self.images[i+1])

            # Initial triangulation
            E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if mask is not None:
                pts1 = pts1[mask.ravel() == 1]
                pts2 = pts2[mask.ravel() == 1]

            if pts1.shape[0] < 4: continue # not enough points to estimate pose
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
            points_3d = self._triangulate(pts1, pts2, self.camera_rotations[-1], self.camera_translations[-1], R, t)
            
            # Use 3d points to refine pose estimation
            _, rvec, t, _ = cv2.solvePnPRansac(points_3d, pts2, self.K, self.dist)
            rvec, t = cv2.solvePnPRefineLM(points_3d, pts2, self.K, self.dist, rvec, t)
            R, _ = cv2.Rodrigues(rvec)
            
            # Use initial guess for nonlinear triangulation
            points_3d = self.nonlinear_triangulation(points_3d, self.camera_rotations[-1], self.camera_translations[-1], R, t, pts1, pts2)

            if i == 0: self.points_2d.append(pts1)
            self.points_2d.append(pts2)
            self.camera_rotations.append(R @ self.camera_rotations[-1])
            self.camera_translations.append(R @ self.camera_translations[-1] + t)
            self.points_3d.append(points_3d)
                    
        return np.vstack(self.points_3d)

if __name__ == "__main__":
    K = np.array([[1.59877590e+03, 0, 5.07401915e+02], [0, 1.57986433e+03, 7.23899817e+02], [0, 0, 1]])
    dist =  np.array([1.01762294e-01, -1.85222000e+00, -1.95598585e-02, -2.43105406e-03, 4.57588149e+00])

    sfm = SfM(SuperPointWrapper(), SuperGlueWrapper(), K, dist)

    result = sfm.reconstruct("benchy")
    plot_3d_points(result)