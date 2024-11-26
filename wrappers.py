from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from SuperGlue.superglue import SuperGlue
import torch
import numpy as np

class SuperPointWrapper:
    def __init__(self, max_keypoints=256):
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
        self.processor.do_resize = False  # Disable resizing
        self.max_keypoints = max_keypoints

    def detect(self, img):
        # Process image using SuperPoint
        inputs = self.processor(img, return_tensors="pt")
        outputs = self.model(**inputs)

        # Extract keypoints and descriptors
        kp = outputs.keypoints[0].detach().numpy()
        desc = outputs.descriptors[0].detach().numpy()
        scores = outputs.scores[0].cpu().detach().numpy()
        
        indices = np.argsort(scores)[::-1]  # Sort in descending order
        indices = indices[:self.max_keypoints]  # Take top N indices
        kp = kp[indices]
        desc = desc[indices]
        scores = scores[indices]

        return kp, desc, scores

class SuperGlueWrapper:
    def __init__(self, weights='indoor', config={}):
        self.config = {**SuperGlue.default_config, **config, 'weights': weights}
        self.superglue = SuperGlue(self.config)
        if torch.cuda.is_available():
            self.superglue.cuda()  # Move the model to the GPU if available
        self.superglue.eval()

    def match(self, keypoints0, descriptors0, keypoints1, descriptors1, scores0, scores1, image0, image1):
        device = next(self.superglue.parameters()).device

        # Convert to tensors and move to the correct device
        kpts0 = torch.from_numpy(keypoints0).float().to(device)[None]
        desc0 = torch.from_numpy(descriptors0).float().to(device)[None]
        scores0 = torch.from_numpy(scores0).float().to(device)[None]
        kpts1 = torch.from_numpy(keypoints1).float().to(device)[None]
        desc1 = torch.from_numpy(descriptors1).float().to(device)[None]
        scores1 = torch.from_numpy(scores1).float().to(device)[None]

        image0 = torch.from_numpy(image0).unsqueeze(0).to(device)  # Add batch dim
        image1 = torch.from_numpy(image1).unsqueeze(0).to(device)  # Add batch dim

        data = {
            'keypoints0': kpts0, 
            'descriptors0': desc0, 
            'keypoints1': kpts1, 
            'descriptors1': desc1,
            'scores0': scores0,
            'scores1': scores1,
            'image0': image0,
            'image1': image1
        }

        pred = self.superglue(data)
        matches0 = pred['matches0'][0].cpu().numpy()
        valid = matches0 > -1
        if not valid.any(): return None, None
        mkpts0 = keypoints0[valid]
        mkpts1 = keypoints1[matches0[valid]]
        return mkpts0, mkpts1