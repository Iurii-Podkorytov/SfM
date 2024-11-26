import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_3d_points(points3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Triangulated 3D Points')
    plt.show(block=True)

def plot_images_inline(images, titles=None, cmap='gray'):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    for i, image in enumerate(images):
        ax = axes[i] if num_images > 1 else axes # Handle single image case
        ax.imshow(image, cmap=cmap) 
        ax.axis('off')  # Turn off axis labels and ticks
        if titles and len(titles) == num_images:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()

def plot_keypoints(image1, kp1, image2, kp2, is_superpoint=False, titles=["Image 1 Keypoints", "Image 2 Keypoints"]):
    if len(image1.shape) == 2:
        img1_bgr = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    else:
        img1_bgr = image1.copy() # Make a copy - It's a good practice

    # Do the same for image2
    if len(image2.shape) == 2:
        img2_bgr = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    else:
        img2_bgr = image2.copy()
    if is_superpoint:
        for x, y in kp1:
            cv2.circle(img1_bgr, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)  # Red filled circles
        for x, y in kp2:
            cv2.circle(img2_bgr, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)
    
    else:
        img1_bgr = cv2.drawKeypoints(image1, kp1, None, color=(0, 255, 0), flags=0)
        img2_bgr = cv2.drawKeypoints(image2, kp2, None, color=(0, 255, 0), flags=0)
    
    plot_images_inline([img1_bgr, img2_bgr], titles=titles)
    plt.show()


def plot_matches(img1, img2, kp1, kp2, matches, mask):
    matched = cv2.drawMatchesKnn(img1, 
                                kp1, 
                                img2, 
                                kp2, 
                                matches,
                                outImg=None, 
                                matchColor=(0, 155, 0), 
                                singlePointColor=(0, 255, 255), 
                                matchesMask=mask, 
                                flags=0) 
    plt.title("Matched Features")
    plt.imshow(matched)
    plt.show()

def plot_super_matches(pts1, pts2, img1, img2):
    # Convert images to BGR if they are grayscale
    if len(img1.shape) == 2:
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_bgr = img1.copy()

    if len(img2.shape) == 2:
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_bgr = img2.copy()

    max_height = max(img1_bgr.shape[0], img2_bgr.shape[0])

    padded_img1 = np.zeros((max_height, img1_bgr.shape[1], 3), dtype=np.uint8) # Assuming 3 channels (BGR)
    padded_img2 = np.zeros((max_height, img2_bgr.shape[1], 3), dtype=np.uint8)


    padded_img1[:img1_bgr.shape[0], : ,:] = img1_bgr
    padded_img2[:img2_bgr.shape[0], : ,:] = img2_bgr

    img_combined = np.hstack([padded_img1, padded_img2])

    # Draw circles and lines
    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i, 0]), int(pts1[i, 1])
        x2, y2 = int(pts2[i, 0]) + img1.shape[1], int(pts2[i, 1])  # Offset x2 by width of img1

        cv2.circle(img_combined, (x1, y1), 3, (0, 0, 255), -1)  # Red circles
        cv2.circle(img_combined, (x2, y2), 3, (0, 0, 255), -1)
        cv2.line(img_combined, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green lines

    # Display the combined image
    plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for Matplotlib
    plt.title("SuperPoint Matches")
    plt.axis('off')  # Hide axis ticks
    plt.show()

def plot_3d_points_and_cameras(points3D, Rs, ts):  # Combined plotting function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot 3D points
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='r', marker='o', label='3D Points')
    # Plot camera poses
    for R, t in zip(Rs, ts):
        ax.scatter(t[0], t[1], t[2], c='b', marker='^', label='Camera') # Camera center
        axes_length = 0.1
        axes = R @ np.eye(3) * axes_length
        for axis in axes:
            ax.plot([t[0], t[0] + axis[0]], [t[1], t[1] + axis[1]], [t[2], t[2] + axis[2]], c='g')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Triangulated 3D Points and Camera Poses')
    plt.show(block=True)