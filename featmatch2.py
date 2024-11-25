import cv2
import numpy as np
import pickle
import argparse
import os
from time import time

# Import transformers for SuperPoint
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from PIL import Image

from utils import *


def FeatMatch(opts, data_files=[]):
    if len(data_files) == 0:
        img_names = sorted(
            [x for x in os.listdir(opts.data_dir) if x.split('.')[-1] in opts.ext],
            key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x
        )
        img_paths = [os.path.join(opts.data_dir, x) for x in img_names]
    else:
        img_paths = data_files
        img_names = sorted(
            [x.split('/')[-1] for x in data_files],
            key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x
        )

    print(f"*****************ft****image names={img_names}")

    feat_out_dir = os.path.join(opts.out_dir, 'features', opts.features)
    matches_out_dir = os.path.join(opts.out_dir, 'matches', opts.matcher)

    if not os.path.exists(feat_out_dir):
        os.makedirs(feat_out_dir)
    if not os.path.exists(matches_out_dir):
        os.makedirs(matches_out_dir)

    data = []
    t1 = time()

    # Initialize the SuperPoint processor and model if required
    if opts.features == "SuperPoint":
        processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        img_name, img_format = img_names[i].split('.')[0], opts.ext[0]
        img = img[:, :, ::-1]  # Convert BGR to RGB

        if opts.features == "SURF":
            feat = cv2.xfeatures2d.SURF_create()
            kp, desc = feat.detectAndCompute(img, None)
            save_path = os.path.join(feat_out_dir, f"feat_img/kp_{img_name}.{img_format}")
            SaveKeypointsImage(img, kp, save_path, is_superpoint=False)
        elif opts.features == "SIFT":
            feat = cv2.SIFT_create()
            kp, desc = feat.detectAndCompute(img, None)
            save_path = os.path.join(feat_out_dir, f"feat_img/kp_{img_name}.{img_format}")
            SaveKeypointsImage(img, kp, save_path, is_superpoint=False)
        elif opts.features == "FAST":
            feat = cv2.FastFeatureDetector_create()
            kp = feat.detect(img, None)
            desc = None
            save_path = os.path.join(feat_out_dir, f"feat_img/kp_{img_name}.{img_format}")
            SaveKeypointsImage(img, kp, save_path, is_superpoint=False)
        elif opts.features == "ORB":
            feat = cv2.ORB_create()
            kp, desc = feat.detectAndCompute(img, None)
            save_path = os.path.join(feat_out_dir, f"feat_img/kp_{img_name}.{img_format}")
            SaveKeypointsImage(img, kp, save_path, is_superpoint=False)
        elif opts.features == "SuperPoint":
            # Process image using SuperPoint
            
            # Modify the processor to avoid resizing
            processor.do_resize = False  # Disable resizing
            # Optionally, you can inspect or override the size parameter:
            # processor.size = None  # This ensures no resizing is applied
            
            inputs = processor(img, return_tensors="pt")
            outputs = model(**inputs)

            # Extract keypoints and descriptors
            kp = outputs.keypoints[0].detach().numpy()  # Keypoints
            desc = outputs.descriptors[0].detach().numpy()  # Descriptors

            # Save image with keypoints visualized
            save_path = os.path.join(feat_out_dir, f"feat_img/kp_{img_name}.{img_format}")
            SaveKeypointsImage(img, kp, save_path, is_superpoint=True)

        else:
            raise ValueError(f"Unknown feature type: {opts.features}")

        data.append((img_name, kp, desc))

        # Serialize keypoints and descriptors
        kp_ = SerializeKeypoints(kp)

        with open(os.path.join(feat_out_dir, f'kp_{img_name}.pkl'), 'wb') as out:
            pickle.dump(kp_, out)

        with open(os.path.join(feat_out_dir, f'desc_{img_name}.pkl'), 'wb') as out:
            pickle.dump(desc, out)

        if opts.save_results:
            raise NotImplementedError

        t2 = time()
        if (i % opts.print_every) == 0:
            print(f'FEATURES DONE: {i + 1}/{len(img_paths)} [time={t2 - t1:.2f}s]')

        t1 = time()

    num_done = 0
    num_matches = ((len(img_paths) - 1) * (len(img_paths))) / 2

    t1 = time()
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            img_name1, kp1, desc1 = data[i]
            img_name2, kp2, desc2 = data[j]

            if opts.matcher == "BFMatcher":
                matcher = cv2.BFMatcher(crossCheck=opts.cross_check)
            elif opts.matcher == "FlannBasedMatcher":
                matcher = cv2.FlannBasedMatcher()
            else:
                raise ValueError(f"Unknown matcher type: {opts.matcher}")

            if desc1 is not None and desc2 is not None:
                matches = matcher.match(desc1, desc2)
                matches = sorted(matches, key=lambda x: x.distance)
                matches_ = SerializeMatches(matches)

                pickle_path = os.path.join(matches_out_dir, f'match_{img_name1}_{img_name2}.pkl')
                try:
                    with open(pickle_path, 'wb') as out:
                        pickle.dump(matches_, out)
                except Exception as e:
                    print(f"Error saving matches to {pickle_path}: {e}")

                # Visualize matches
                # show_matched_points(img_paths[i], img_paths[j], kp1, kp2, matches)

            num_done += 1
            t2 = time()

            if (num_done % opts.print_every) == 0:
                print(f'MATCHES DONE: {num_done}/{num_matches} [time={t2 - t1:.2f}s]')

            t1 = time()
    print_arguments(vars(opts))

def SetArguments(parser):
    # directories stuff
    parser.add_argument('--data_files', action='store', type=str, default='', dest='data_files')
    parser.add_argument('--data_dir', action='store', type=str, default='../data/duck/images/',
                        dest='data_dir', help='directory containing images (default: ../data/\
                        duck/images/)') 
    parser.add_argument('--ext', action='store', type=str, default='jpg', dest='ext',
                        help='comma separated string of allowed image extensions \
                        (default: jpg)')
    parser.add_argument('--out_dir', action='store', type=str, default='../data/duck/',
                        dest='out_dir', help='root directory to store results in \
                        (default: ../data/duck)')

    # feature matching args
    parser.add_argument('--features', action='store', type=str, default='SIFT', dest='features',
                        help='[SIFT|SURF|ORB|FAST|SuperPoint] Feature algorithm to use (default: SIFT)')
    parser.add_argument('--matcher', action='store', type=str, default='BFMatcher', dest='matcher',
                        help='[BFMatcher|FlannBasedMatcher] Matching algorithm to use \
                        (default: BFMatcher)')
    parser.add_argument('--cross_check', action='store', type=bool, default=True, dest='cross_check',
                        help='[True|False] Whether to cross check feature matching or not \
                        (default: True)')

    # misc
    parser.add_argument('--print_every', action='store', type=int, default=1, dest='print_every',
                        help='[1,+inf] print progress every print_every seconds, -1 to disable \
                        (default: 1)')
    parser.add_argument('--save_results', action='store', type=str, default=False,
                        dest='save_results', help='[True|False] whether to save images with\
                        keypoints drawn on them (default: False)')


def PostprocessArgs(opts):
    opts.ext = opts.ext.split(',')
    opts.data_files = opts.data_files.split(',') if opts.data_files else []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    PostprocessArgs(opts)

    FeatMatch(opts, opts.data_files)
