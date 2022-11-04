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

import ipdb as pdb

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

rospy.init_node('visualizer-3dod-dataset-reduced')

# dogma_dataset_full_path = os.path.join('./3dod-grids-full', 'data')
dogma_dataset_reduced_path = os.path.join('./3dod-grids-reduced', 'data')
velodyne_dataset_reduced_path = os.path.join('./3dod-pointclouds-reduced', 'data')

kitti_3dod_path = os.path.join('../kitti-3dod','training')



def visualize_rviz(state_grid, pointcloud, cv_image, P):
	print('visualization')
	this_time = rospy.Time()

	# Setup the tfs
	header = Header()
	header.stamp = this_time
	header.frame_id = base_link_frame

	tf_msg = TFMessage()

	# Tf velodyne
	tf_velo = TransformStamped()
	tf_velo.header.frame_id = base_link_frame
	tf_velo.child_frame_id = velo_frame
	tf_velo.transform.translation.x = 0.0
	tf_velo.transform.translation.y = 0.0
	tf_velo.transform.translation.z = 1.733
	tf_velo.transform.rotation.x = 0.0
	tf_velo.transform.rotation.y = 0.0
	tf_velo.transform.rotation.z = 0.0
	tf_velo.transform.rotation.w = 1.0

	tf_msg.transforms.append(tf_velo)

	# Tf camera
	tf_camera = TransformStamped()
	tf_camera.header.frame_id = base_link_frame
	tf_camera.child_frame_id = camera_frame
	tf_camera.transform.translation.x = 0.273
	tf_camera.transform.translation.y = 0.060
	tf_camera.transform.translation.z = 1.661
	tf_camera.transform.rotation.x = -0.495
	tf_camera.transform.rotation.y = 0.5
	tf_camera.transform.rotation.z = -0.5
	tf_camera.transform.rotation.w = 0.505

	tf_msg.transforms.append(tf_camera)

	pub_tf_static.publish(tf_msg)
	pub_clock.publish(this_time)

	# Pointcloud message
	header = Header()
	header.stamp = this_time
	header.frame_id = velo_frame
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


	# camera message
	calib = CameraInfo()
	calib.header.frame_id = camera_frame
	calib.header.stamp = this_time
	# calib.P = util['P_rect_02']
	calib.P = P
	calib.height, calib.width = cv_image.shape[:2]
	
	encoding = "bgr8"
	image_message = CvBridge().cv2_to_imgmsg(cv_image, encoding=encoding)
	image_message.header.frame_id = camera_frame
	image_message.header.stamp = this_time

	pub_camera.publish(image_message)
	pub_camera_info.publish(calib)
	pub_clock.publish(this_time)



if __name__ == '__main__':

	# we iterate over all the grid files in dogma_dataset_reduced_path (state and velocity)
	# for each of them, we get the name
	# load state_grid_reduced and pointcloud_reduced
	# load calibration and image in kitti_3dod_path

	# visualize in rviz

	if os.path.exists(dogma_dataset_reduced_path):
		grid_files = sorted([f for f in os.listdir(dogma_dataset_reduced_path)
			if os.path.isfile(os.path.join(dogma_dataset_reduced_path, f))])
	else:
		raise FileNotFoundError(
			errno.ENOENT, os.strerror(errno.ENOENT), dogma_dataset_reduced_path)

	for f in grid_files:
		if f[7:12] == 'state':
			name = f[0:6]

			state_grid_reduced_path = os.path.join(dogma_dataset_reduced_path, name + '_state_grid_reduced.npy')
			if not os.path.exists(state_grid_reduced_path):
				raise FileNotFoundError(
					errno.ENOENT, os.strerror(errno.ENOENT), state_grid_reduced_path)

			calibration_file_path = os.path.join(kitti_3dod_path, 'calib', '{:06d}'.format(int(name)) + '.txt')
			if not os.path.exists(calibration_file_path):
				raise FileNotFoundError(
					errno.ENOENT, os.strerror(errno.ENOENT), calibration_file_path)

			pointcloud_reduced_path = os.path.join(velodyne_dataset_reduced_path, '{:06d}'.format(int(name)) + '_reduced.npy')
			if not os.path.exists(pointcloud_reduced_path):
				raise FileNotFoundError(
					errno.ENOENT, os.strerror(errno.ENOENT), pointcloud_reduced_path)

			image_file_path = os.path.join(kitti_3dod_path, 'image_2', '{:06d}'.format(int(name)) + '.png')
			if not os.path.exists(image_file_path):
				raise FileNotFoundError(
					errno.ENOENT, os.strerror(errno.ENOENT), image_file_path)

			this_state_grid = np.load(state_grid_reduced_path)
			num_point_features = 4
			this_pointcloud = np.fromfile(pointcloud_reduced_path, dtype=np.float32, count=-1).reshape([-1, num_point_features])
			cv_image = cv2.imread(image_file_path)
			print('Grid, pointcloud and image data for sample {} loaded'.format(name))

			with open(calibration_file_path, 'r') as f:
				cal_lines = f.readlines()
			P = [float(info) for info in cal_lines[2].split(' ')[1:13]]

			# visualize on rviz
			visualize_rviz(this_state_grid, this_pointcloud, cv_image, P)

			pdb.set_trace()