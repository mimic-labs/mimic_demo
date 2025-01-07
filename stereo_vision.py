import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d
matplotlib.rcParams.update({'font.size': 20})

import sys
import glob
sys.path.append('./third_party/RAFT-Stereo/')
sys.path.append('./third_party/RAFT-Stereo/core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt

DEVICE = 'cuda'


def findObjImgPoints(calibrationDir, checkerboard=(7,10), draw=True):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    row, col = checkerboard
     
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((row*col,3), np.float32)
    objp[:,:2] = np.mgrid[0:row,0:col].T.reshape(-1,2)
     
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(calibrationDir+'/*.JPG')
    print(images)
    for fname in images:
        img = cv2.imread(fname)
        r,w,_ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray, (int(w*0.25),int(r*0.25)))
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (row,col), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            out = cv2.drawChessboardCorners(img, (row,col), corners2, ret)
            corners2_reshaped = corners2.reshape(-1, 2)
            if draw:
                plt.scatter(corners2_reshaped[:, 0], corners2_reshaped[:, 1], c='r', s=10)
                plt.imshow(out)
                plt.show()
    return objpoints, imgpoints, gray.shape

def undistortImg(img, origmtx, dist, newcameramtx, roi, crop=True):
    """
    img: cv2 image
    crop: should return crop to ROI (region of interest) or not
    """
    dst = cv2.undistort(img, origmtx, dist, None, newcameramtx)
    
    # crop the image
    if crop:
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
    
    return dst

def getExtrinsics(imgPath, origmtx, dist, checkerboard=(7,10)):
    """
    checkerboard: provide with (rows, cols)
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
    row, col = checkerboard
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    objp = np.zeros((row*col,3), np.float32)
    objp[:,:2] = np.mgrid[0:row,0:col].T.reshape(-1,2)
    # Find the rotation and translation vectors.
    ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, origmtx, dist)
    rotation, _ = cv2.Rodrigues(rvecs)
    return rotation, tvecs

def rotation_matrix_from_euler(angles: tuple) -> np.ndarray:
    """
    Computes the rotation matrix from Euler angles (in radians).
    """
    rx, ry, rz = angles

    # Rotation matrix around the x-axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    # Rotation matrix around the y-axis
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    # Rotation matrix around the z-axis
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    # Combine the rotations: R = Rz * Ry * Rx
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    return R

def compute_relative_extrinsics(R1: tuple, T1: np.ndarray, R2: tuple, T2: np.ndarray) -> tuple:
    """
    Computes the relative extrinsic parameters (rotation and translation) between two cameras.
    """

    R_relative = np.dot(R2, np.linalg.inv(R1))
    T_relative = T2 - R_relative @ T1

    return R_relative, T_relative

def buildQ(K, T_relative):
    return np.array([[1.0, 0.0, 0.0, -float(K[0][-1])],
                     [0.0, 1.0, 0.0, -float(K[1][-1])],
                     [0.0, 0.0, 0.0, -float(K[0][0])],
                     [0.0, 0.0, -1/np.linalg.norm(T_relative), 0.0]])

#test
objpoints, imgpoints, shapeOfCalibImg = findObjImgPoints("calibrationPhotos", draw=False)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shapeOfCalibImg[::-1], None, None)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )

h, w = shapeOfCalibImg[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

R1_w2c, T1_w2c = getExtrinsics("leftreal.JPG", mtx, dist)
R2_w2c, T2_w2c = getExtrinsics("rightreal.JPG", mtx, dist)

img_lefto = cv2.imread("leftreal.JPG")
img_left = undistortImg(img_lefto, mtx, dist, newcameramtx, roi)

img_right = cv2.imread("rightreal.JPG")
img_right = undistortImg(img_right, mtx, dist, newcameramtx, roi)

Kl = newcameramtx
Kr = newcameramtx
Dl = np.zeros(5)
Dr = np.zeros(5)
T1 = T1_w2c
T2 = T2_w2c
R1o = R1_w2c
R2o = R2_w2c

R, T = compute_relative_extrinsics(R1o, T1, R2o, T2)
R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(Kl, Dl, Kr, Dr, img_left.shape[:2][::-1], R, T)
xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, img_left.shape[:2][::-1], cv2.CV_32FC1)
xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, img_left.shape[:2][::-1], cv2.CV_32FC1)
left_img_rectified = cv2.remap(img_left, xmap1, ymap1, cv2.INTER_LINEAR)
right_img_rectified = cv2.remap(img_right, xmap2, ymap2, cv2.INTER_LINEAR)
################ The above is to create rectified images so it can be fed into our disparity predictor ################

class DepthCalculation:
    def __init__(self):
        self.net = None
        self.pc_world = None
        self.pc_cam = None

    def load_model(self):
        self.net = self.initRAFTModel("cuda:0")

    def preprocess(self, img1, img2):
        image1, origshape = self.load_image(img1)
        image2, origshape = self.load_image(img2)
        
        # TODO 2: Put this into a function
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        return image1, image2

    def postprocess(self, flow_up):
        return -1.0*flow_up.detach().cpu().numpy().squeeze()

    def compute_disparity(self, image1, image2):
        img1, img2 = self.preprocess(image1,image2)
        
        with torch.no_grad():
            _, flow_up = model(img1, img2, iters=32, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()
        
        disparity = self.postprocess(flow_up)
        return disparity

    def getPointCloud(self, disparity_map, Q, rectifyRotation, Rw2c, Tw2c, weird=False):
        pixel_wise_disparities = disparity_map
        points_3D = cv2.reprojectImageTo3D(pixel_wise_disparities, Q)
        if rectifyRotation is not None:
            realPoints3dcam = np.linalg.inv(rectifyRotation) @ points_3D[:,:,:,None]
        else: 
            realPoints3dcam = points_3D
    
        if weird:
            realPoints3d = Rw2c @ realPoints3dcam + Tw2c # rotation_matrix_from_euler(R1_angles) @ realPoints3dcam + T1 #wtf is happening, wait nvm this may be right
        else: 
            realPoints3d = np.linalg.inv(Rw2c) @ (realPoints3dcam - Tw2c) # rotation_matrix_from_euler(R1_angles) @ realPoints3dcam + T1 #wtf is happening, wait nvm this may be right
        self.pc_world = realPoints3d
        self.pc_cam = realPoints3dcam
        return realPoints3d, realPoints3dcam
    
    def getXYZWorld(self, pixel):
        x,y = pixel
        return self.pc_world[y,x]

    def getXYZCam(self, pixel):
        x,y = pixel
        return self.pc_cam[y,x]

    def load_image(self, imfile, resize=None):
        # img = np.array(Image.open(imfile)).astype(np.uint8)
        # img = torch.from_numpy(img).permute(2, 0, 1).float()
        # return img[None].to(DEVICE)
        if isinstance(imfile, str):
            origimg = cv2.imread(imfile)
        else:
            origimg = imfile
        img = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)
        if resize:
            img = cv2.resize(img, resize, cv2.INTER_AREA)
        # -> C,H,W
        # Normalization done in the model itself.
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE), origimg.shape

    def initRAFTModel(self, device):
        parser = argparse.ArgumentParser()
        parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
        parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
        parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
        parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
        parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
        
        # Architecture choices
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
        parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
        parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
        parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
        parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
        parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
        argss = "--restore_ckpt /nethome/abati7/flash/Work/mimic/third_party/RAFT-Stereo/models/raftstereo-middlebury.pth --corr_implementation alt --mixed_precision"
        args = parser.parse_args(argss.split(" "))
        
        model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        model.load_state_dict(torch.load(args.restore_ckpt))
        
        model = model.module
        model.to(device)
        model.eval()
        return model

depthCalc = DepthCalculation()
depthCalc.load_model()
disparity = depthCalc.compute_disparity(left_img_rectified,right_img_rectified)
realPoints3dWorldView, realPoints3dCamView = depthCalc.getPointCloud(disparity, Q, R1, R1o, T1)


# model = initRAFTModel()

# image1, origshape = load_image("leftstereo.png")
# image2, origshape = load_image("rightstereo.png")

# # TODO 2: Put this into a function
# padder = InputPadder(image1.shape, divis_by=32)
# image1, image2 = padder.pad(image1, image2)
# with torch.no_grad():
#     _, flow_up = model(image1, image2, iters=32, test_mode=True)
# flow_up = padder.unpad(flow_up).squeeze()
# flow_up = -flow_up.detach().cpu().numpy().squeeze()

# realPoints3dWorldView, realPoints3dCamView = getPointCloud(flow_up, Q, R1, R1o, T1)

points_3D = realPoints3dWorldView.reshape((-1,3))
# points_3D = realPoints3dCamView.reshape((-1,3))


# TODO 3: Put this into a function for saving point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3D)

output_filename = "pccam.ply"
o3d.io.write_point_cloud(output_filename, pcd)