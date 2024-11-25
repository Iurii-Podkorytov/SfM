import numpy as np 
import cv2 
import argparse
import pickle
import os 
from time import time
import matplotlib.pyplot as plt

from utils import * 
import pdb 

class Camera(object): 
    def __init__(self, R, t, ref): 
        self.R = R 
        self.t = t 
        self.ref = ref

class Match(object): 
    def __init__(self, matches, img1pts, img2pts, img1idx, img2idx, mask): 
        self.matches = matches
        self.img1pts, self.img2pts = img1pts, img2pts 
        self.img1idx, self.img2idx = img1idx, img2idx
        self.mask = mask

class SFM(object): 
    def __init__(self, opts): 
        self.opts = opts
        self.point_cloud = np.zeros((0,3))

        #setting up directory stuff..
        self.images_dir = os.path.join(opts.data_dir,opts.dataset, 'images')
        self.feat_dir = os.path.join(opts.data_dir, opts.dataset, 'features', opts.features)
        self.matches_dir = os.path.join(opts.data_dir, opts.dataset, 'matches', opts.matcher)
        self.out_cloud_dir = os.path.join(opts.out_dir, opts.dataset, 'point-clouds')
        self.out_err_dir = os.path.join(opts.out_dir, opts.dataset, 'errors')

        #output directories
        if not os.path.exists(self.out_cloud_dir): 
            os.makedirs(self.out_cloud_dir)

        if (opts.plot_error is True) and (not os.path.exists(self.out_err_dir)): 
            os.makedirs(self.out_err_dir)

        self.image_names = [
            x.split('.')[0] for x in sorted(
                os.listdir(self.images_dir),
                key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x
            )
            if x.split('.')[-1] in opts.ext
        ]

        self.image_format = opts.ext[0]
        #setting up shared parameters for the pipeline
        self.image_data, self.matches_data, errors = {}, {}, {}
        
        if opts.matcher == "BFMatcher":
            self.matcher = cv2.BFMatcher(crossCheck=opts.cross_check)
        elif opts.matcher == "FlannBasedMatcher":
            # FLANN matcher does not support crossCheck, initialize without it
            self.matcher = cv2.FlannBasedMatcher()
        else:
            raise ValueError(f"Unknown matcher type: {opts.matcher}")

        if opts.calibration_mat == 'benchmark': 
            self.K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]])
        elif opts.calibration_mat == 'lg_g3': 
            self.K = np.array([[3.97*320, 0, 320],[0, 3.97*320, 240],[0,0,1]])
        elif opts.calibration_mat == 's9':
            self.K = np.array( [[ 1.03825261e+03,  0.00000000e+00,  2.47798585e+02],
                                [ 0.00000000e+00,  1.03797300e+03, -9.01702837e+02],
                                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
            
            self.K = np.array([[3.29422290e+03, 0.00000000e+00, 1.98917924e+03],
                                [0.00000000e+00, 4.42531898e+03, 1.51004887e+03],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        elif opts.calibration_mat == 'yura':
            self.K = np.array([[1.65541048e+03, 0.00000000e+00, 4.46578657e+02],
                                [0.00000000e+00, 1.61896963e+03, 7.08040616e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        elif opts.calibration_mat == 'easy':
            self.K = np.array([[689.87, 0, 380.17],
                               [0, 691.04, 251.70],
                               [0, 0, 1]])
        elif opts.calibration_mat == 's92':
            self.K = np.array([[129.78111875,   0.        ,  80.97475988],
                                [  0.        , 129.74617625,  62.28691862],
                                [  0.        ,   0.        ,   1.        ]])
        else: 
            raise NotImplementedError
        
    def _LoadFeatures(self, name):

        feature_path = os.path.join(self.feat_dir, f'kp_{name}.pkl')
        desc_path = os.path.join(self.feat_dir, f'desc_{name}.pkl')
        kp, desc = None, None

        try:
            with open(feature_path, 'rb') as f:
                kp = DeserializeKeypoints(pickle.load(f))
        except (EOFError, FileNotFoundError):
            print(f"Error loading keypoints from {feature_path}")
        
        try:
            with open(desc_path, 'rb') as f:
                desc = pickle.load(f)
        except (EOFError, FileNotFoundError):
            print(f"Error loading descriptors from {desc_path}")

        return kp, desc

    def _LoadMatches(self, name1, name2): 
        with open(os.path.join(self.matches_dir,'match_{}_{}.pkl'.format(name1,name2))
                    ,'rb') as f: 
            matches = pickle.load(f)
        matches = DeserializeMatches(matches)
        return matches

    def _GetAlignedMatches(self,kp1,desc1,kp2,desc2,matches):
        img1idx = np.array([m.queryIdx for m in matches])
        img2idx = np.array([m.trainIdx for m in matches])

        #filtering out the keypoints that were matched. 
        kp1_ = (np.array(kp1))[img1idx]
        kp2_ = (np.array(kp2))[img2idx]

        #retreiving the image coordinates of matched keypoints
        img1pts = np.array([kp.pt for kp in kp1_])
        img2pts = np.array([kp.pt for kp in kp2_])

        return img1pts, img2pts, img1idx, img2idx

    def _BaselinePoseEstimation(self, name1, name2):

        kp1, desc1 = self._LoadFeatures(name1)
        kp2, desc2 = self._LoadFeatures(name2)

        matches = self._LoadMatches(name1, name2)
        matches = sorted(matches, key = lambda x:x.distance)

        img1pts, img2pts, img1idx, img2idx = self._GetAlignedMatches(kp1,desc1,kp2,
                                                                    desc2,matches)
        
        # F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=opts.fund_method,
                                        # param1=opts.outlier_thres,param2=opts.fund_prob)
        
        F, mask = cv2.findFundamentalMat(
            img1pts, 
            img2pts, 
            method=opts.fund_method,
            # method=getattr(cv2, opts.fund_method),  # Dynamic method selection
            ransacReprojThreshold=opts.outlier_thres,
            confidence=opts.fund_prob
        )

        mask = mask.astype(bool).flatten()

        E = self.K.T.dot(F.dot(self.K))
        _,R,t,_ = cv2.recoverPose(E,img1pts[mask],img2pts[mask],self.K)

        self.image_data[name1] = [np.eye(3,3), np.zeros((3,1)), np.ones((len(kp1),))*-1]
        self.image_data[name2] = [R,t,np.ones((len(kp2),))*-1]

        self.matches_data[(name1,name2)] = [matches, img1pts[mask], img2pts[mask], 
                                            img1idx[mask],img2idx[mask]]

        return R,t

    def _TriangulateTwoViews(self, name1, name2): 

        def __TriangulateTwoViews(img1pts, img2pts, R1, t1, R2, t2): 
            img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
            img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]

            img1ptsNorm = (np.linalg.inv(self.K).dot(img1ptsHom.T)).T
            img2ptsNorm = (np.linalg.inv(self.K).dot(img2ptsHom.T)).T

            img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
            img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

            pts4d = cv2.triangulatePoints(np.hstack((R1,t1)),np.hstack((R2,t2)),
                                            img1ptsNorm.T,img2ptsNorm.T)
            pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]

            return pts3d

        def _Update3DReference(ref1, ref2, img1idx, img2idx, upp_limit, low_limit=0): 

            ref1[img1idx] = np.arange(upp_limit) + low_limit
            ref2[img2idx] = np.arange(upp_limit) + low_limit

            return ref1, ref2

        R1, t1, ref1 = self.image_data[name1]
        R2, t2, ref2 = self.image_data[name2]

        _, img1pts, img2pts, img1idx, img2idx = self.matches_data[(name1,name2)]
        
        new_point_cloud = __TriangulateTwoViews(img1pts, img2pts, R1, t1, R2, t2)
        self.point_cloud = np.concatenate((self.point_cloud, new_point_cloud), axis=0)

        ref1, ref2 = _Update3DReference(ref1, ref2, img1idx, img2idx,new_point_cloud.shape[0],
                                        self.point_cloud.shape[0]-new_point_cloud.shape[0])
        self.image_data[name1][-1] = ref1 
        self.image_data[name2][-1] = ref2 

    def _TriangulateNewView(self, name):

        prev_name = list(self.image_data.keys())[-2]

        kp1, desc1 = self._LoadFeatures(prev_name)
        kp2, desc2 = self._LoadFeatures(name)
        print(f"555555555555555555555: cur={name}, prev={prev_name}, {self.image_data.keys()}")  

        prev_name_ref = self.image_data[prev_name][-1]
        matches = self._LoadMatches(prev_name,name)
        matches = [match for match in matches if prev_name_ref[match.queryIdx] < 0]

        if len(matches) > 7:
            show_matched_points(f"{self.images_dir}/{name}.jpg", f"{self.images_dir}/{prev_name}.jpg",
                                kp1, kp2, matches, save=True, feat_det=opts.features, feat_match=opts.matcher)
            
            matches = sorted(matches, key = lambda x:x.distance)

            img1pts, img2pts, img1idx, img2idx = self._GetAlignedMatches(kp1,desc1,kp2,
                                                                        desc2,matches)
            
            # F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=opts.fund_method,
            #                                 param1=opts.outlier_thres,
            #                                 param2=opts.fund_prob)
            
            F, mask = cv2.findFundamentalMat(
                img1pts, 
                img2pts, 
                method=opts.fund_method,
                # method=getattr(cv2, opts.fund_method),  # Dynamic method selection
                ransacReprojThreshold=opts.outlier_thres,
                confidence=opts.fund_prob
            )
            # print(F, mask)

            if mask is None:
                print(f"Warning: Mask is None, skipping this pair. {name}")
                print(f"matches = {len(matches)}, kp1={len(kp1)}, kp2={len(kp2)}, desc1={len(desc1)}, desc2={len(desc2)}")

                # continue  # Skip this iteration if mask is None
            # print(f"========matches = {len(matches)}, kp1={len(kp1)}, kp2={len(kp2)}, desc1={len(desc1)}, desc2={len(desc2)}")
            mask = mask.astype(bool).flatten()

            self.matches_data[(prev_name,name)] = [matches, img1pts[mask], img2pts[mask], 
                                        img1idx[mask],img2idx[mask]]
            self._TriangulateTwoViews(prev_name, name)

        else: 
            print('skipping {} and {}, matches={}'.format(prev_name, name, len(matches)))
        
        # for prev_name in self.image_data.keys(): 
        #     if prev_name != name: 
        #         kp1, desc1 = self._LoadFeatures(prev_name)
        #         kp2, desc2 = self._LoadFeatures(name)
        #         print(f"555555555555555555555: cur={name}, prev={prev_name}, {self.image_data.keys()}")  

        #         prev_name_ref = self.image_data[prev_name][-1]
        #         matches = self._LoadMatches(prev_name,name)
        #         matches = [match for match in matches if prev_name_ref[match.queryIdx] < 0]

        #         if len(matches) > 7:
        #             show_matched_points(f"{self.images_dir}/{name}.jpg", f"{self.images_dir}/{prev_name}.jpg", kp1, kp2, matches) 
        #             matches = sorted(matches, key = lambda x:x.distance)

        #             img1pts, img2pts, img1idx, img2idx = self._GetAlignedMatches(kp1,desc1,kp2,
        #                                                                         desc2,matches)
                    
        #             # F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=opts.fund_method,
        #             #                                 param1=opts.outlier_thres,
        #             #                                 param2=opts.fund_prob)
                    
        #             F, mask = cv2.findFundamentalMat(
        #                 img1pts, 
        #                 img2pts, 
        #                 method=opts.fund_method,
        #                 # method=getattr(cv2, opts.fund_method),  # Dynamic method selection
        #                 ransacReprojThreshold=opts.outlier_thres,
        #                 confidence=opts.fund_prob
        #             )
        #             # print(F, mask)

        #             if mask is None:
        #                 print(f"Warning: Mask is None, skipping this pair. {name}")
        #                 print(f"matches = {len(matches)}, kp1={len(kp1)}, kp2={len(kp2)}, desc1={len(desc1)}, desc2={len(desc2)}")

        #                 continue  # Skip this iteration if mask is None
        #             # print(f"========matches = {len(matches)}, kp1={len(kp1)}, kp2={len(kp2)}, desc1={len(desc1)}, desc2={len(desc2)}")
        #             mask = mask.astype(bool).flatten()

        #             self.matches_data[(prev_name,name)] = [matches, img1pts[mask], img2pts[mask], 
        #                                         img1idx[mask],img2idx[mask]]
        #             self._TriangulateTwoViews(prev_name, name)

        #         else: 
        #             print('skipping {} and {}, matches={}'.format(prev_name, name, len(matches)))
        
    def _NewViewPoseEstimation(self, name): 
        
        def _Find2D3DMatches(): 
            
            matcher_temp = getattr(cv2, opts.matcher)()
            kps, descs = [], []
            for n in self.image_names: 
                if n in self.image_data.keys():
                    kp, desc = self._LoadFeatures(n)

                    kps.append(kp)
                    descs.append(desc)
            
            matcher_temp.add(descs)
            matcher_temp.train()

            kp, desc = self._LoadFeatures(name)

            matches_2d3d = matcher_temp.match(queryDescriptors=desc)

            #retrieving 2d and 3d points
            pts3d, pts2d = np.zeros((0,3)), np.zeros((0,2))
            for m in matches_2d3d: 
                train_img_idx, desc_idx, new_img_idx = m.imgIdx, m.trainIdx, m.queryIdx
                point_cloud_idx = self.image_data[self.image_names[train_img_idx]][-1][desc_idx]
                
                #if the match corresponds to a point in 3d point cloud
                if point_cloud_idx >= 0: 
                    new_pt = self.point_cloud[int(point_cloud_idx)]
                    pts3d = np.concatenate((pts3d, new_pt[np.newaxis]),axis=0)

                    new_pt = np.array(kp[int(new_img_idx)].pt)
                    pts2d = np.concatenate((pts2d, new_pt[np.newaxis]),axis=0)

            return pts3d, pts2d, len(kp)
        
        def __Find2D3DMatches():
            pts3d, pts2d = np.zeros((0,3)), np.zeros((0,2))
            kp, desc = self._LoadFeatures(name)

            i = 0 
            
            while i < len(self.image_names): 
                curr_name = self.image_names[i]

                if curr_name in self.image_data.keys(): 
                    matches = self._LoadMatches(curr_name, name)
                    print(f"================{matches}")
                    ref = self.image_data[curr_name][-1]
                    pts3d_idx = np.array([ref[m.queryIdx] for m in matches \
                                        if ref[m.queryIdx] > 0])
                    pts2d_ = np.array([kp[m.trainIdx].pt for m in matches \
                                        if ref[m.queryIdx] > 0])
                                        
                    pts3d = np.concatenate((pts3d, self.point_cloud[pts3d_idx.astype(int)]),axis=0)
                    pts2d = np.concatenate((pts2d, pts2d_),axis=0)

                i += 1 

            return pts3d, pts2d, len(kp)

        pts3d, pts2d, ref_len = _Find2D3DMatches()
        _, R, t, _ = cv2.solvePnPRansac(pts3d[:,np.newaxis],pts2d[:,np.newaxis],self.K,None,
                            confidence=self.opts.pnp_prob,flags=getattr(cv2,self.opts.pnp_method),
                            reprojectionError=self.opts.reprojection_thres)
        R,_=cv2.Rodrigues(R)
        self.image_data[name] = [R,t,np.ones((ref_len,))*-1]

    def ToPly(self, filename):
        
        def _GetColors(): 
            colors = np.zeros_like(self.point_cloud)
            
            for k in self.image_data.keys(): 
                _, _, ref = self.image_data[k]
                kp, desc = self._LoadFeatures(k)
                kp = np.array(kp)[ref>=0]
                image_pts = np.array([_kp.pt for _kp in kp])
                image = cv2.imread(os.path.join(self.images_dir, k+f'.{self.image_format}'))[:,:,::-1]
                colors[ref[ref>=0].astype(int)] = image[image_pts[:,1].astype(int),
                                                        image_pts[:,0].astype(int)]
            
            return colors

        colors = _GetColors()
        pts2ply(self.point_cloud, colors, filename)

    def _ComputeReprojectionError(self, name): 
        
        def huber_loss(diff, delta=1.0):
            """
            Compute the Huber Loss for a set of differences.
            
            Args:
                diff (numpy.ndarray): Difference between points (img_pts - reproj_pts).
                delta (float): Threshold for switching between quadratic and linear loss.

            Returns:
                float: The Huber Loss.
            """
            sq_diff = np.sum(diff**2, axis=-1)  # Squared differences
            abs_diff = np.sqrt(sq_diff)         # Euclidean distances (L2 norm)
            
            # Apply Huber Loss formula
            loss = np.where(
                abs_diff <= delta,
                0.5 * sq_diff,                 # Quadratic region
                delta * (abs_diff - 0.5 * delta)  # Linear region
            )
            return np.mean(loss)

        def log_error(img_pts, reproj_pts, delta=1.0):
            """
            Compute the log-based loss for reprojection error.

            Args:
                img_pts (ndarray): Ground truth image points.
                reproj_pts (ndarray): Reprojected points from the 3D structure.
                delta (float): Scale parameter for controlling robustness (default: 1.0).

            Returns:
                float: Logarithmic loss value.
            """
            residuals = np.linalg.norm(img_pts - reproj_pts, axis=-1)  # L2 residuals
            loss = np.log(1 + (residuals**2) / (delta**2))
            return np.mean(loss)

        def _ComputeReprojections(X,R,t,K): 
            outh = K.dot(R.dot(X.T) + t )
            out = cv2.convertPointsFromHomogeneous(outh.T)[:,0,:]
            return out

        R, t, ref = self.image_data[name]
        reproj_pts = _ComputeReprojections(self.point_cloud[ref[ref>0].astype(int)], R, t, self.K)

        kp, desc = self._LoadFeatures(name)
        img_pts = np.array([kp_.pt for i, kp_ in enumerate(kp) if ref[i] > 0])
        

        # Compute the error
        # diff = img_pts - reproj_pts
        # err = huber_loss(diff, delta=1e-100)  # Set `delta` as per your dataset
        # assert np.all(np.isfinite(self.point_cloud)), f"Point cloud contains invalid values!{name}"

        delta = 0.0001  # Set based on your data (can be tuned)
        # err = log_error(img_pts, reproj_pts, delta=delta)

        err = np.mean(np.sqrt(np.sum((img_pts-reproj_pts)**2,axis=-1)))

        print(f"Reprojection error: {err}")

        if math.isnan(err):
            print(f"img pts {name} = {img_pts}")
            print(f"reproj pts {name} = {reproj_pts}")
            print(f"x = {self.point_cloud[ref[ref>0].astype(int)]}")
            print(f"R = {R}")
            print(f"t = {t}")
            print(f"k = {self.K}")


        if self.opts.plot_error: 
            fig,ax = plt.subplots()
            image = cv2.imread(os.path.join(self.images_dir, name+'.jpg'))[:,:,::-1]
            ax = DrawCorrespondences(image, img_pts, reproj_pts, ax)
            
            ax.set_title('reprojection error = {}'.format(err))

            fig.savefig(os.path.join(self.out_err_dir, '{}.png'.format(name)))
            plt.close(fig)
            
        return err
        
    def Run(self):
        name1, name2 = self.image_names[0], self.image_names[1]
        total_time, errors = 0, []

        t1 = time()
        self._BaselinePoseEstimation(name1, name2)
        t2 = time()
        this_time = t2-t1
        total_time += this_time
        print('Baseline Cameras {0}, {1}: Pose Estimation [time={2:.3}s]'.format(name1, name2,
                                                                                 this_time))
        print(f"name1={name1} name2={name2}")
        print(self.image_names)
        self._TriangulateTwoViews(name1, name2)
        t1 = time()
        this_time = t1-t2
        total_time += this_time
        print('Baseline Cameras {0}, {1}: Baseline Triangulation [time={2:.3}s]'.format(name1, 
                                                                                name2, this_time))

        views_done = 2 

        #3d point cloud generation and reprojection error evaluation
        self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_{}_view.ply'.format(views_done)))

        err1 = self._ComputeReprojectionError(name1)
        err2 = self._ComputeReprojectionError(name2)
        errors.append(err1)
        errors.append(err2)

        print('Camera {}: Reprojection Error = {}'.format(name1, err1))
        print('Camera {}: Reprojection Error = {}'.format(name2, err2))

        for new_name in self.image_names[2:]: 

            #new camera registration
            t1 = time()
            self._NewViewPoseEstimation(new_name)
            t2 = time()
            this_time = t2-t1
            total_time += this_time
            print('Camera {0}: Pose Estimation [time={1:.3}s]'.format(new_name, this_time))

            #triangulation for new registered camera
            self._TriangulateNewView(new_name)
            t1 = time()
            this_time = t1-t2
            total_time += this_time
            print('Camera {0}: Triangulation [time={1:.3}s]'.format(new_name, this_time))

            #3d point cloud update and error for new camera
            views_done += 1 
            self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_{}_view.ply'.format(views_done)))

            new_err = self._ComputeReprojectionError(new_name)
            errors.append(new_err)
            print('Camera {}: Reprojection Error = {}'.format(new_name, new_err))

        mean_error = sum(errors) / float(len(errors))
        print('Reconstruction Completed: Mean Reprojection Error = {2} [t={0:.6}s], \
                Results stored in {1}'.format(total_time, self.opts.out_dir, mean_error))
        
    def printargs(self):
        print_arguments(vars(self.opts))

def SetArguments(parser): 

    #directory stuff
    parser.add_argument('--data_dir',action='store',type=str,default='data/',dest='data_dir',
                        help='root directory containing input data (default: data/)') 
    parser.add_argument('--dataset',action='store',type=str,default='duck',dest='dataset',
                        help='name of dataset (default: duck)') 
    parser.add_argument('--ext',action='store',type=str,default='jpg',dest='ext', 
                        help='comma seperated string of allowed image extensions \
                        (default: jpg)') 
    parser.add_argument('--out_dir',action='store',type=str,default='results/',dest='out_dir',
                        help='root directory to store results in (default: results/)') 

    #matching parameters
    parser.add_argument('--features',action='store',type=str,default='SIFT',dest='features',
                        help='[SIFT|SURF|ORB|FAST|SuperPoint] Feature algorithm to use (default: SIFT)')
    parser.add_argument('--matcher',action='store',type=str,default='BFMatcher',dest='matcher',
                        help='[BFMatcher|FlannBasedMatcher] Matching algorithm to use \
                        (default: BFMatcher)') 
    parser.add_argument('--cross_check',action='store',type=bool,default=True,dest='cross_check',
                        help='[True|False] Whether to cross check feature matching or not \
                        (default: True)') 

    #epipolar geometry parameters
    parser.add_argument('--calibration_mat',action='store',type=str,default='s9',
                        dest='calibration_mat',help='[benchmark|lg_g3|s9|yura|easy] type of intrinsic camera \
                        to use (default: s9)')
    parser.add_argument('--fund_method',action='store',type=str,default='FM_RANSAC',
                        dest='fund_method',help='method to estimate fundamental matrix \
                        (default: FM_RANSAC)')
    parser.add_argument('--outlier_thres',action='store',type=float,default=.9,
                        dest='outlier_thres',help='threhold value of outlier to be used in\
                         fundamental matrix estimation (default: 0.9)')
    parser.add_argument('--fund_prob',action='store',type=float,default=.9,dest='fund_prob',
                        help='confidence in fundamental matrix estimation required (default: 0.9)')
    
    #PnP parameters
    parser.add_argument('--pnp_method',action='store',type=str,default='SOLVEPNP_DLS',
                        dest='pnp_method',help='[SOLVEPNP_DLS|SOLVEPNP_EPNP|..] method used for\
                        PnP estimation, see OpenCV doc for more options (default: SOLVEPNP_DLS')
    parser.add_argument('--pnp_prob',action='store',type=float,default=.99,dest='pnp_prob',
                        help='confidence in PnP estimation required (default: 0.99)')
    parser.add_argument('--reprojection_thres',action='store',type=float,default=8.,
                        dest='reprojection_thres',help='reprojection threshold in PnP estimation \
                        (default: 8.)')

    #misc
    parser.add_argument('--plot_error',action='store',type=bool,default=False,dest='plot_error')

def PostprocessArgs(opts): 
    opts.fund_method = getattr(cv2,opts.fund_method)
    opts.ext = opts.ext.split(',')

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    PostprocessArgs(opts)
    
    sfm = SFM(opts)
    sfm.Run()
    sfm.printargs()