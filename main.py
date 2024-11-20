import os
import cv2
import numpy as np

class SfM:
    def __init__(self, feature_detector, matcher, F_estimator, triangulator, bundle_adjuster):
        self.feature_detector = feature_detector
        self.matcher = matcher
        self.F_estimator = F_estimator
        self.triangulator = triangulator
        self.bundle_adjuster = bundle_adjuster

    def reconstruct(self, images_folder):
        # Step 1: Extract features with feature_detector

        # Step 2: Match features with matcher

        # Step 3: Estimate fundamental matrix

        # Step 4: Find valid camera poses

        # Step 5: Triangulation

        # Step 6: Bundle adjustment

        return points_3d
