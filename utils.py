import numpy as np 
import cv2 
import pdb
import os
from PIL import Image
import argparse
import math


def show_matched_points(img1_path, img2_path, kp1, kp2, matches, save=False, feat_det="SIFT", feat_match="BFMatcher"):
    """
    Display or save matched keypoints between two images, including image names and the number of matches.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        kp1 (list): Keypoints of the first image.
        kp2 (list): Keypoints of the second image.
        matches (list): List of DMatch objects.
        save (bool): Whether to save the output image instead of displaying it.
        save_path (str): Path to save the output image if `save` is True.
    """
    print(f"Image 1 Path: {img1_path}")
    print(f"Image 2 Path: {img2_path}")

    # Load the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Check if images are loaded successfully
    if img1 is None:
        raise FileNotFoundError(f"Image 1 not found or could not be loaded: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Image 2 not found or could not be loaded: {img2_path}")

    # Draw matches
    matches_to_draw = matches[:]
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches_to_draw, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Get image names
    img1_name = os.path.basename(img1_path)
    img2_name = os.path.basename(img2_path)

    # Prepare text for the image
    text = f"{img1_name} <=> {img2_name} | Matches: {len(matches_to_draw)} | Feat_det: {feat_det} | Feat_matcher: {feat_match}"

    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = 10  # Margin from the left
    text_y = text_size[1] + 10  # Margin from the top

    # Add a background rectangle for text
    cv2.rectangle(
        matched_img,
        (text_x - 5, text_y - text_size[1] - 5),
        (text_x + text_size[0] + 5, text_y + 5),
        (0, 0, 0),
        thickness=-1
    )

    # Add the text on top of the rectangle
    cv2.putText(
        matched_img,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA
    )

    if save:
        # Extract save directory from img1_path
        save_dir = os.path.dirname(img1_path).replace("images", "features")
        os.makedirs(save_dir, exist_ok=True)

        # Construct save path
        save_path = os.path.join(save_dir, feat_det, feat_match, f"matched_{img1_name}")

        # Save the output image
        res = cv2.imwrite(save_path, matched_img)
        print(f"Matched points image saved to {save_path}, res={res}")
    else:
        # Display the matched image
        cv2.imshow("Matched Points", matched_img)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

def print_arguments(args_dict):
    """
    Prints the parsed arguments in a simple, formatted way.

    Args:
        args_dict: Dictionary of parsed arguments.
    """
    print("\n================= ARGUMENTS =================")
    for key, value in args_dict.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print("============================================\n")


def SaveKeypointsImage(image, keypoints, save_path, is_superpoint=False):
    """
    Draws detected keypoints on the image and saves the result.

    Args:
        image (numpy.ndarray): Input image (HxWxC) in RGB format.
        keypoints (numpy.ndarray or list[cv2.KeyPoint]): 
            Keypoints detected by SuperPoint or OpenCV.
        save_path (str): Path to save the image with keypoints drawn.
        is_superpoint (bool): True if keypoints are from SuperPoint, False if from OpenCV.

    Returns:
        None
    """
    # Convert image to BGR format for OpenCV visualization
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw keypoints on the image
    if is_superpoint:
        for kp in keypoints:
            x, y = kp[0], kp[1]
            cv2.circle(image_bgr, (int(x), int(y)), radius=3, color=(0, 0, 0), thickness=-1)
    else:
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])  # Extract (x, y) from cv2.KeyPoint
            cv2.circle(image_bgr, (x, y), radius=3, color=(0, 0, 0), thickness=-1)  # Blue dots for keypoints

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the image with keypoints
    cv2.imwrite(save_path, image_bgr)

def SerializeKeypoints(kp): 
    """Serialize list of keypoint objects or NumPy array so it can be saved using pickle.

    Args:
        kp: List of OpenCV KeyPoint objects or a NumPy array of keypoints (SuperPoint format).

    Returns:
        out: Serialized list of keypoint objects.
    """
    out = []
    if isinstance(kp, np.ndarray):  # Handle SuperPoint keypoints
        for kp_ in kp:
            temp = (tuple(kp_), 0, 0, 0, 0, -1)  # Default values for other attributes
            out.append(temp)
    else:  # Handle OpenCV KeyPoint objects
        for kp_ in kp:
            temp = (kp_.pt, kp_.size, kp_.angle, kp_.response, kp_.octave, kp_.class_id)
            out.append(temp)
    return out

def DeserializeKeypoints(kp): 
    """Deserialize list of keypoint objects so it can be converted back to
    native opencv's format.
    
    Args: 
    kp: List of serialized keypoint objects 
    
    Returns: 
    out: Deserialized list of keypoint objects"""

    # out = []
    # for point in kp:
    #     temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2],
    #      _response=point[3], _octave=point[4], _class_id=point[5]) 
    #     out.append(temp)

    # return out

    deserialized_kps = []
    for point in kp:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2],
                            response=point[3], octave=point[4], class_id=point[5])
        deserialized_kps.append(temp)
    return deserialized_kps

def SerializeMatches(matches): 
    """Serializes dictionary of matches so it can be saved using pickle
    
    Args: 
    matches: List of matches object
    
    Returns: 
    out: Serialized list of matches object"""

    out = []
    for match in matches: 
        matchTemp = (match.queryIdx, match.trainIdx, match.imgIdx, match.distance) 
        out.append(matchTemp)
    return out

def DeserializeMatches(matches): 
    """Deserialize dictionary of matches so it can be converted back to 
    native opencv's format. 
    
    Args: 
    matches: Serialized list of matches object
    
    Returns: 
    out: List of matches object"""

    out = []
    for match in matches:
        out.append(cv2.DMatch(match[0],match[1],match[2],match[3])) 
    return out

def GetAlignedMatches(kp1,desc1,kp2,desc2,matches):
    """Aligns the keypoints so that a row of first keypoints corresponds to the same row 
    of another keypoints
    
    Args: 
    kp1: List of keypoints from first (left) image
    desc1: List of desciptros from first (left) image
    kp2: List of keypoints from second (right) image
    desc2: List of desciptros from second (right) image
    matches: List of matches object
    
    Returns: 
    img1pts, img2pts: (n,2) array where img1pts[i] corresponds to img2pts[i] 
    """

    #Sorting in case matches array isn't already sorted
    matches = sorted(matches, key = lambda x:x.distance)

    #retrieving corresponding indices of keypoints (in both images) from matches.  
    img1idx = np.array([m.queryIdx for m in matches])
    img2idx = np.array([m.trainIdx for m in matches])

    #filtering out the keypoints that were NOT matched. 
    kp1_ = (np.array(kp1))[img1idx]
    kp2_ = (np.array(kp2))[img2idx]

    #retreiving the image coordinates of matched keypoints
    img1pts = np.array([kp.pt for kp in kp1_])
    img2pts = np.array([kp.pt for kp in kp2_])

    return img1pts,img2pts

def pts2ply(pts,colors,filename='out.ply'): 
    """Saves an ndarray of 3D coordinates (in meshlab format)"""

    with open(filename,'w') as f: 
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(pts.shape[0]))
        
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        
        f.write('end_header\n')
        
        colors = colors.astype(int)
        for pt, cl in zip(pts, colors): 
            f.write('{} {} {} {} {} {}\n'.format(pt[0], pt[1], pt[2],
                                                cl[0], cl[1], cl[2]))


def DrawCorrespondences(img, ptsTrue, ptsReproj, ax, drawOnly=50): 
    """
    Draws correspondence between ground truth and reprojected feature point

    Args: 
    ptsTrue, ptsReproj: (n,2) numpy array
    ax: matplotlib axis object
    drawOnly: max number of random points to draw

    Returns: 
    ax: matplotlib axis object
    """
    ax.imshow(img)
    
    randidx = np.random.choice(ptsTrue.shape[0],size=(drawOnly,),replace=False)
    ptsTrue_, ptsReproj_ = ptsTrue[randidx], ptsReproj[randidx]
    
    colors = colors=np.random.rand(drawOnly,3)
    
    ax.scatter(ptsTrue_[:,0],ptsTrue_[:,1],marker='x',c='r',linewidths=.1, label='Ground Truths')
    ax.scatter(ptsReproj_[:,0],ptsReproj_[:,1],marker='x',c='b',linewidths=.1, label='Reprojected')
    ax.legend()

    return ax