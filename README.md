# mimic_demo

## Starter Guide
```bash
#for linux systems

#general deps
conda create -n mim python=3.12

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia 

pip3 install opencv-python matplotlib open3d opt_einsum

# sam2 deps
cd third_party/sam2
pip3 install -e ".[notebooks]"
cd checkpoints
./download_ckpts.sh

# raft-stereo deps
cd third_part/RAFT-Stereo
./download_models.sh
```

## Using stereo_vision.py

For further information, check the file. There are 4 steps that are useful for calculating intrinsic matrix, undistorting images, stereorectifying, calculating extrinsics, and building Q matrix.

```python
# NOTE: All that is needed for this is Q, Rotation_w2c, R_stereorectify (optional), Translation_w2c
depthCalc = DepthCalculation()
depthCalc.load_model()
disparity = depthCalc.compute_disparity(left_img_rectified,right_img_rectified)
realPoints3dWorldView, realPoints3dCamView = depthCalc.getPointCloud(disparity, Q, R1, R1o, T1)

# NOTE: realPoints3dWorldView is a mapping from undistorted image coords to 3d world coords
# SO we would use: depthCalc.getXYZWorld(pixelXY)


# NOTE: Visualization stuff
points_3D = realPoints3dWorldView.reshape((-1,3))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3D)

output_filename = "pccam.ply"
o3d.io.write_point_cloud(output_filename, pcd)
```


## Using matching.py

```python
from matching import get_mask1_bestmask2, get_corresponding_contacts

sam2, mask_predictor, prompted_predictor = initialize_sam2("cuda:0")
dinov2 = initialize_dinov2("cuda:0")

image_path1 = "images/query_img.jpg"
image_path2 = "images/current_vid_0.jpg"
pos_points = torch.tensor([[[750, 850], [1000,800]]]) # [B,N,2]

resultsIm1, resultsIm2 = get_mask1_bestmask2((dinov2, mask_predictor, prompted_predictor), image_path1, image_path2, pos_points)

get_corresponding_contacts(resultsIm1, resultsIm2, pos_points.tolist())

# NOTE: resultsIm2.orig_image_space_coords = where matched point is in original image space

# NOTE: for visualization: resultsIm2.image_with_contact
```