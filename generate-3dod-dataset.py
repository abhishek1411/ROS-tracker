import ipdb as pdb
import numpy as np
import os

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

rospy.init_node('3dod_dataset_creator')


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
	# train_rand contains comma separated values indicating for each line (line 1 is image 0 in the 3d dataset) which line (1-indexed) in train_mapping corresponds to in the raw dataset
	with open('train_rand.txt', 'r') as f:
		lines = f.readlines()

	odidx_to_mapidx = lines[0].split(',') # example: odidx_to_mapidx[0] = '7282'
	# example, image 0 in the dataset has the mapping to raw dataset described in line 7282 from train_mapping

	odidx_to_date_drive_sample = [] # row is 3d-kitti image index, columns are date, drive, and sample

	log_file_grids_not_found_path = os.path.join('.', '3dod-grids-full', 'grids_not_found.log')
	if os.path.exists(log_file_grids_not_found_path):
		os.remove(log_file_grids_not_found_path)

	nr_missed_grids = 0
	nr_grids_dataset = 0

	with open('train_mapping.txt', 'r') as f:
		lines = f.readlines() # example line '2011_09_26 2011_09_26_drive_0005_sync 0000000109\n'
		for img_idx in range(len(odidx_to_mapidx)):
			map_idx = int(odidx_to_mapidx[img_idx]) - 1 # 0-indexed
			infos = lines[map_idx].split(' ')
			date = infos[0]
			drive = infos[1][17:21]
			sample = infos[2][:-1] # remove \n

			# img_idx = 52
			# map_idx = 161  
			# date = '2011_09_26' 
			# drive = '0009'
			# sample = '0000000218'


			# if date == '2011_09_26' and drive == '0009':
			if True:
				print('im: {} map: {}  Date: {}  Drive: {}  Sample: {}'.format(img_idx, map_idx, date, drive, sample))

				# fix alignment problem for 2011_09_26_drive_0009, pointcloud missing msgs in bag, after 0000000176.bin and before 0000000181.bin)
				# however, grids were named in sequence, without taking into account this gap
				# That is, grid 177 corresponds to pointcloud 181
				
				if date == '2011_09_26' and drive == '0009' and int(sample)>=181:
					grid_sample = '{:010d}'.format(int(sample) - 4)
				else:
					grid_sample = sample

				# find the grids
				state_grid_path = os.path.join('.', 'grids', date, drive, grid_sample + '_state_grid.npy')
				velocity_grid_path = os.path.join('.', 'grids', date, drive, grid_sample + '_velocity_grid.npy')


				state_grid_found = False
				velocity_grid_found = False
				pointcloud_found = False
				image_found = False
				calibration_file_found = False

				if os.path.exists(state_grid_path):
					# print('state_grid_found')
					state_grid_found = True
				else:
					print('*** State grid not found! im: {} date: {} drive: {} sample: {}***'.format(img_idx, date, drive, sample))
					pdb.set_trace()

				if os.path.exists(velocity_grid_path):
					# print('velocity_grid_found')
					velocity_grid_found = True
				else:
					print('*** Velocity grid not found! im: {} date: {} drive: {} sample: {}***'.format(img_idx, date, drive, sample))
					pdb.set_trace()

				if not state_grid_found or not velocity_grid_found:
					# add img_idx to grids_not_found.log
					with open(log_file_grids_not_found_path,"a+") as f:
						f.write('{:010d} {} {} {}'.format(img_idx,date,drive,sample) + '\n')
					nr_missed_grids += 1
				
				# find velodyne pointcloud
				# pointcloud_path = os.path.join('.', 'extracted', date, drive, date, date + '_drive_' + drive + '_sync',
				# 	'velodyne_points', 'data', sample + '.bin')

				pointcloud_path = os.path.join('../kitti-3dod/training/velodyne/', '{:06d}'.format(img_idx) + '.bin')
				if os.path.exists(pointcloud_path):
					pointcloud_found = True
				else:
					print('*** Pointcloud not found! im: {} date: {} drive: {} sample: {}***'.format(img_idx, date, drive, sample))
					pdb.set_trace()

				# find corresponding left color image (left camera is the reference camera)
				# image_path = os.path.join('.', 'extracted', date, drive, date, date + '_drive_' + drive + '_sync',
				# 	'image_02', 'data', sample + '.png')
				image_path = os.path.join('../kitti-3dod/training/image_2/', '{:06d}'.format(img_idx) + '.png')
				if os.path.exists(image_path):
					image_found = True
				else:
					print('*** Image not found! im: {} date: {} drive: {} sample: {}***'.format(img_idx, date, drive, sample))
					pdb.set_trace()

				# find calibration file
				# calibration_file_path = os.path.join('.', 'extracted', date, 'calib_cam_to_cam.txt')
				calibration_file_path = os.path.join('../kitti-3dod/training/calib/', '{:06d}'.format(img_idx) + '.txt')

				if os.path.exists(calibration_file_path):
					calibration_file_found = True
				else:
					print('*** Calibration file not found! im: {} date: {} drive: {} sample: {}***'.format(img_idx, date, drive, sample))
					pdb.set_trace()


				# Load the data, visualize in rviz, and copy the grids to the folder of the 3dod-dogma dataset with
				# the sample image naming convention
				if state_grid_found and velocity_grid_found and pointcloud_found and image_found and calibration_file_found:
					# load data
					this_state_grid = np.load(state_grid_path)
					this_velocity_grid = np.load(velocity_grid_path)
					# this_pointcloud = np.load(pointcloud_path)
					num_point_features = 4
					this_pointcloud = np.fromfile(pointcloud_path, dtype=np.float32, count=-1).reshape([-1, num_point_features])

					cv_image = cv2.imread(image_path)

					# print('Grid, pointcloud and image data for img {} loaded'.format(img_idx))

					# util = pykitti.utils.read_calib_file(calibration_file_path)
					# load P from calibration file
					with open(calibration_file_path, 'r') as f:
						cal_lines = f.readlines()
					P = [float(info) for info in cal_lines[2].split(' ')[1:13]]

					# visualize on rviz
					visualize_rviz(this_state_grid, this_pointcloud, cv_image, P)

					# copy state and velocity grids to the folder of the 3dod-dogma dataset with the index sample
					dogma_dataset_path = os.path.join('.', '3dod-grids-full', 'data')
					if os.path.exists(dogma_dataset_path):
						copyfile(state_grid_path, os.path.join(dogma_dataset_path, '{:010d}'.format(img_idx) + '_state_grid.npy'))
						copyfile(velocity_grid_path, os.path.join(dogma_dataset_path, '{:010d}'.format(img_idx) + '_velocity_grid.npy'))
						print('Grids copied to folder {} for sample {:010d}'.format(dogma_dataset_path, img_idx))
						nr_grids_dataset +=1
						print('Total grids added: {}'.format(nr_grids_dataset))


				print('\n')

				pdb.set_trace()


	# Print total number of grids not found
	print('##### SUMMARY #####')
	print('Total number of grids in the dataset: {}'.format(nr_grids_dataset))
	print('Total number of grids missed: {}'.format(nr_missed_grids))
	print('List of missed grids:')

	# Load the infos
	with open(log_file_grids_not_found_path,'r') as f:
		lines_missed = f.readlines()

	info_missed = []
	for l in lines_missed:
		info_missed.append(l.split())

	# Print list sorted by date, drive, sample
	info_missed = sorted(info_missed, key=operator.itemgetter(1,2,3))
	for l in info_missed:
		print(l)

	pdb.set_trace()

