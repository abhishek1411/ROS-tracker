import ipdb as pdb
import numpy as np
import os
import errno

import rospy
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
import sensor_msgs.point_cloud2 as pcl2
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform, Point32
from rosgraph_msgs.msg import Clock

from e_motion_perception_msgs.msg import FloatOccupancyGrid, VelocityGrid

from cv_bridge import CvBridge
import cv2
import pykitti

from shutil import copyfile
import operator

from second.core import box_np_ops

import ipdb as pdb

from tqdm import tqdm


###########
# TO BE RUN WITH POINTP CONDA ENVIRONMENT
###########


velo_topic = '/kitti/velo/pointcloud'
tf_topic = '/tf'
tfs_topic = '/tf_static'
clock_topic = '/clock'   
camera_topic = '/kitti/camera_color/left/image_rect_color'
camera_info_topic = '/kitti/camera_color/left/camera_info'
state_grid_topic = '/kitti/state_grid'

base_link_frame = '/kitti/base_link'
velo_frame = '/kitti/velo_link'
camera_frame = '/camera_color_left'

pub_velo = rospy.Publisher(velo_topic, PointCloud2, queue_size=10)
pub_tf_static = rospy.Publisher(tfs_topic, TFMessage, queue_size=10)
pub_state_grid = rospy.Publisher(state_grid_topic, FloatOccupancyGrid, queue_size=10)
pub_camera = rospy.Publisher(camera_topic, Image, queue_size=10)
pub_camera_info = rospy.Publisher(camera_info_topic, CameraInfo, queue_size=10)
pub_clock = rospy.Publisher(clock_topic, Clock, queue_size=10)

rospy.init_node('3dod_dataset_reducer')

dogma_dataset_full_path = os.path.join('./3dod-grids-full', 'data')
dogma_dataset_reduced_path = os.path.join('./3dod-grids-reduced-white', 'data')
velodyne_dataset_reduced_path = os.path.join('./3dod-pointclouds-reduced', 'data')

kitti_3dod_path = os.path.join('../kitti-3dod','training')



def _extend_matrix(mat):
	mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
	return mat

def visualize_rviz_fix(state_grid, pointcloud):
	print('visualization')
	this_time = rospy.Time()

	# Pointcloud message
	header = Header()
	header.stamp = this_time
	header.frame_id = base_link_frame
	pointcloud_msg = pcl2.create_cloud_xyz32(header, pointcloud[:,0:3])
	pub_velo.publish(pointcloud_msg)
	pub_clock.publish(this_time)

	# state_grid message
	state_grid_msg = FloatOccupancyGrid()
	header = Header()
	header.stamp = this_time
	header.frame_id = base_link_frame
	state_grid_msg.header = header

	state_grid_msg.info.resolution = 0.16
	state_grid_msg.info.width = state_grid.shape[1]
	state_grid_msg.info.height = state_grid.shape[0]
	state_grid_msg.info.origin.position.y = -39.68
	state_grid_msg.info.origin.orientation.w = 1.0
	state_grid_msg.nb_channels = 4

	data = np.zeros((state_grid.shape[0]*state_grid.shape[1]*4))
	data[0::4] = state_grid[...,0].ravel()
	data[1::4] = state_grid[...,1].ravel()
	data[3::4] = state_grid[...,2].ravel()
	data[2::4] = state_grid[...,3].ravel()
	state_grid_msg.data = data.tolist()

	pub_state_grid.publish(state_grid_msg)
	pub_clock.publish(this_time)



if __name__ == '__main__':

	# we iterate over all the grid files in dogma_dataset_path (state and velocity)
	# for each of them, we get the name
	# load state_grid and velocity_grid
	# load calibration_file, pointcloud and image in kitti_3dod_path

	# reduce pointcloud
	# reduce grids
	# publish reduced pointcloud and state_grid, image
	# copy reduced state and velocity grids to dogma_dataset_path_reduced

	if os.path.exists(dogma_dataset_full_path):
		grid_files = sorted([f for f in os.listdir(dogma_dataset_full_path)
			if os.path.isfile(os.path.join(dogma_dataset_full_path, f))])
	else:
		raise FileNotFoundError(
			errno.ENOENT, os.strerror(errno.ENOENT), dogma_dataset_full_path)

	for f_idx in tqdm(range(len(grid_files))):
		f = grid_files[f_idx]
		if f[11:16] == 'state':
			name = f[0:10]

			state_grid_path = os.path.join(dogma_dataset_full_path, name + '_state_grid.npy')
			if not os.path.exists(state_grid_path):
				raise FileNotFoundError(
					errno.ENOENT, os.strerror(errno.ENOENT), state_grid_path)

			velocity_grid_path = os.path.join(dogma_dataset_full_path, name + '_velocity_grid.npy')
			if not os.path.exists(velocity_grid_path):
				raise FileNotFoundError(
					errno.ENOENT, os.strerror(errno.ENOENT), velocity_grid_path)

			calibration_file_path = os.path.join(kitti_3dod_path, 'calib', '{:06d}'.format(int(name)) + '.txt')
			if not os.path.exists(calibration_file_path):
				raise FileNotFoundError(
					errno.ENOENT, os.strerror(errno.ENOENT), calibration_file_path)

			pointcloud_file_path = os.path.join(kitti_3dod_path, 'velodyne', '{:06d}'.format(int(name)) + '.bin')
			if not os.path.exists(pointcloud_file_path):
				raise FileNotFoundError(
					errno.ENOENT, os.strerror(errno.ENOENT), pointcloud_file_path)

			image_file_path = os.path.join(kitti_3dod_path, 'image_2', '{:06d}'.format(int(name)) + '.png')
			if not os.path.exists(image_file_path):
				raise FileNotFoundError(
					errno.ENOENT, os.strerror(errno.ENOENT), image_file_path)

			this_state_grid = np.load(state_grid_path)
			this_velocity_grid = np.load(velocity_grid_path)
			num_point_features = 4
			this_pointcloud = np.fromfile(pointcloud_file_path, dtype=np.float32, count=-1).reshape([-1, num_point_features])
			cv_image = cv2.imread(image_file_path)
			img_shape = np.array(cv_image.shape[:2], dtype=np.int32)
			print('Grid, pointcloud and image data for sample {} loaded'.format(name))


			extend_matrix = True
			with open(calibration_file_path, 'r') as f:
				lines = f.readlines()

			P2 = np.array(
				[float(info) for info in lines[2].split(' ')[1:13]]).reshape(
					[3, 4])
			if extend_matrix:
				P2 = _extend_matrix(P2)

			R0_rect = np.array([
				float(info) for info in lines[4].split(' ')[1:10]
			]).reshape([3, 3])
			if extend_matrix:
				rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
				rect_4x4[3, 3] = 1.
				rect_4x4[:3, :3] = R0_rect
			else:
				rect_4x4 = R0_rect
			rect = rect_4x4

			Tr_velo_to_cam = np.array([
				float(info) for info in lines[5].split(' ')[1:13]
			]).reshape([3, 4])
			if extend_matrix:
				Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
			Trv2c = Tr_velo_to_cam

			this_pointcloud_reduced = box_np_ops.remove_outside_points(this_pointcloud, rect, Trv2c,
				P2, img_shape)

			# Reduce the grids
			# create a list of point coordinates, each being at the center of each cell at velodyne height
			# height = 0 in velodyne frame
			grid_resolution = 0.16
			offset = np.array([[0.0, 39.68]])
			ncols, nrows = this_state_grid.shape[0:2]
			r, c = np.meshgrid(range(1,nrows+1),range(1,ncols+1))
			cell_center_indeces = np.array([[x,y] for x,y in zip(r.ravel(),c.ravel())], dtype=int) 
			
			cell_center_locations = (cell_center_indeces * grid_resolution) - offset
			# add height
			cell_center_locations = np.hstack((cell_center_locations,np.zeros((cell_center_locations.shape[0],1))))

			# filter
			cell_center_locations_reduced, indeces_kept = box_np_ops.remove_outside_points_indeces(cell_center_locations, rect, Trv2c, P2, img_shape)

			# get back the state_grid_indeces
			# cell_center_indices_reduced =  (cell_center_locations_reduced[:,0:2] + offset) / grid_resolution
			# cell_center_indices_reduced = cell_center_indices_reduced.astype(int)
			# cell_center_indices_reduced = cell_center_indices_reduced[:, [1,0]] - 1

			cell_center_indeces_reduced = cell_center_indeces[indeces_kept.squeeze()] - 1

			this_state_grid_reduced_vis = np.ones(np.shape(this_state_grid))
			this_state_grid_reduced_vis[cell_center_indeces_reduced[:,1],cell_center_indeces_reduced[:,0],:] = \
			this_state_grid[cell_center_indeces_reduced[:,1],cell_center_indeces_reduced[:,0],:]

			# # visualize on rviz
			# visualize_rviz_fix(this_state_grid_reduced_vis, this_pointcloud_reduced)
			# pdb.set_trace()

			# save reduced pointclouds and grids
			# this_state_grid_reduced_saving = np.zeros(np.shape(this_state_grid),dtype=np.float32)
			# this_state_grid_reduced_saving[cell_center_indeces_reduced[:,1],cell_center_indeces_reduced[:,0],:] = \
			# this_state_grid[cell_center_indeces_reduced[:,1],cell_center_indeces_reduced[:,0],:]

			# this_velocity_grid_reduced_saving = np.zeros(np.shape(this_velocity_grid),dtype=np.float32)
			# this_velocity_grid_reduced_saving[cell_center_indeces_reduced[:,1],cell_center_indeces_reduced[:,0],:] = \
			# this_velocity_grid[cell_center_indeces_reduced[:,1],cell_center_indeces_reduced[:,0],:]

			np.save(os.path.join(dogma_dataset_reduced_path,
				'{:06d}_state_grid_reduced'.format(int(name))), this_state_grid_reduced_vis)

			# np.save(os.path.join(dogma_dataset_reduced_path,
			# 	'{:06d}_velocity_grid_reduced'.format(int(name))), this_velocity_grid_reduced_saving)

			# np.save(os.path.join(velodyne_dataset_reduced_path,
			# 	'{:06d}_reduced'.format(int(name))), this_pointcloud_reduced)			

			print('Reduced velodyne and grids saved for sample: {:06d}\n'.format(int(name)))
			
			# pdb.set_trace()