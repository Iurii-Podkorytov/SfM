# Structure from Motion (SfM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)
![Jupyter Notebookdf](https://img.shields.io/badge/jupyter-%23FA0F00.svg?&logo=jupyter&logoColor=white)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

## 📖 Project Overview

This project demonstrates **Structure from Motion (SfM)**, a computer vision technique that reconstructs 3D structures from a sequence of 2D images or video frames. The goal is to generate a 3D point cloud representing the scene, leveraging feature detection, matching, and optimization techniques.

Our approach uses modern feature detectors like **SuperPoint**, feature matchers such as **FlannBasedMatcher**, and robust optimization techniques to refine 3D reconstruction. By calibrating the camera and processing the images, we create point clouds that visualize the scene's spatial structure.

---

## 🛠️ Approach

1. **Feature Extraction**: Detect keypoints in images using detectors like **SIFT** or **SuperPoint**.
2. **Feature Matching**: Match corresponding keypoints between consecutive images using **BFMatcher** or **FlannBasedMatcher**.
3. **Camera Calibration**: Use the provided calibration matrix for precise measurements.
4. **Structure Reconstruction**: Solve for camera poses and triangulate points to generate the 3D structure.
5. **Optimization  (Optional)**: Refine the reconstruction using techniques like bundle adjustment for accuracy.

---

## 🔍 Example Results

Below are some examples in the procces:

![Example 1](https://github.com/Iurii-Podkorytov/SfM/blob/1f8019ae49a6755dfdde4683302c3c11616ef7bf/report%20images/duck_sift.png)

*Feature detection of a toy duck using SIFT*

![Example 2](https://github.com/Iurii-Podkorytov/SfM/blob/1f8019ae49a6755dfdde4683302c3c11616ef7bf/report%20images/duck_sp_bfm.png)  
*Feature matching of a toy duck using BFM*

---

## 🚀 How to Use

1. Shoot a video of an object or take pictures of it.
2. Follow the instructions in the *presentation.ipynb* file

## ⚙️ Libraries and Tools Used

![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?&logo=opencv&logoColor=white) For image processing, feature detection, and matching.

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?&logo=numpy&logoColor=white) For numerical operations and data handling.

![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?&logo=Matplotlib&logoColor=black) For visualizing features and results.

![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?&logo=scipy&logoColor=%white) Used for least squares optimization.
