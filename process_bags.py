#!/usr/bin/env python
import os.path

import rospy
from roslib import message
import rosbag
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, CameraInfo

import os
from os.path import isfile, join, exists

import ipdb as pdb
import time

import numpy as np
import copy
import ctypes
import struct
import pcl
import ros_numpy
from numpy_ros import to_numpy
import sensor_msgs.point_cloud2 as pc2
# import open3d as o3d


def lidar_msg_createpcl(ros_point_cloud,dir,filename):
	xyz = np.array([[0,0,0]])
	rgb = np.array([[0,0,0]])
	#self.lock.acquire()
	gen = pc2.read_points(ros_point_cloud, skip_nans=True)
	int_data = list(gen)

	for x in int_data:
		test = x[3]
		# cast float32 to int so that bitwise operations are possible
		s = struct.pack('>f' ,test)
		i = struct.unpack('>l',s)[0]
		# you can get back the float value by the inverse operations
		pack = ctypes.c_uint32(i).value
		r = (pack & 0x00FF0000)>> 16
		g = (pack & 0x0000FF00)>> 8
		b = (pack & 0x000000FF)
		# prints r,g,b values in the 0-255 range
					# x,y,z can be retrieved from the x[0],x[1],x[2]
		xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
		rgb = np.append(rgb,[[r,g,b]], axis = 0)

	p = pcl.PointCloud(xyz)
	# out_pcd = o3d.geometry.PointCloud()
	# out_pcd.points = o3d.utility.Vector3dVector(xyz)
	# out_pcd.colors = o3d.utility.Vector3dVector(rgb)
	# o3d.io.write_point_cloud(f"{dir}/{filename}.ply",out_pcd)

def callback_1(ros_cloud,dir,filename):
	""" Converts a ROS PointCloud2 message to a pcl PointXYZRGB

	        Args:
	            ros_cloud (PointCloud2): ROS PointCloud2 message

	        Returns:
	            pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
	    """
	points_list = []

	for data in pc2.read_points(ros_cloud, skip_nans=True):
		points_list.append([data[0], data[1], data[2], data[3]])

	pcl_data = pcl.create_xyzi(points_list)
	pcl.io.save_pcd(f"{dir}/{filename}.ply",pcl_data)

	# pc = ros_numpy.numpify(data)
	# pc = to_numpy(data)
	# points=np.zeros((pc.shape[0],3))
	# points[:,0]=pc['x']
	# points[:,1]=pc['y']
	# points[:,2]=pc['z']
	# p = pcl.PointCloud(np.array(points, dtype=np.float32))



if __name__ == '__main__':

	bags_dir = '/home/Abhishek/Desktop/connect_8tb/zoe_bags'

	# velo_topic = 'lidar_top'
	velo_topic = '/zoe/velodyne_points'
	tf_topic = '/tf'
	# clock_topic = '/clock'   
	
	# image_topic = 'cam_front/raw'
	image_topic = '/zoe/camera_front/camera_out/image_rect'

	# camera_info_topic = ''     

	rospy.init_node('data_saver')

	pub_pointcloud = rospy.Publisher(velo_topic, PointCloud2, queue_size=10)
	pub_tf = rospy.Publisher(tf_topic, TFMessage, queue_size=10)
	# pub_tf_static = rospy.Publisher(tfs_topic, TFMessage, queue_size=10)
	# pub_clock = rospy.Publisher(clock_topic, Clock, queue_size=10)
	pub_image = rospy.Publisher(image_topic, Image, queue_size=10)
	# pub_camera_info = rospy.Publisher(camera_info_topic, CameraInfo, queue_size=10)
	# bag_files = sorted([f for f in listdir(bags_dir) if isfile(join(bags_dir, f))])
	bag_files = ['2022_Apr_12-16_00_59.bag']

	for bag_name in bag_files:
		print('Processing bag: {}'.format(bag_name))
		folder_name = bag_name.replace('.bag', '')

		# create folders for the grids, if they do not exist
		this_pcl_path = '/home/Abhishek/Desktop/connect_8tb/zoe_pointcloud'
		this_pcl_path = os.path.join(this_pcl_path,folder_name)
		os.makedirs(this_pcl_path, exist_ok=True)

		bag_path = join(bags_dir, bag_name)
		this_bag = rosbag.Bag(bag_path)

		print(this_bag)
		print('REMEMBER TO RESTART CMCDOT')

		# pdb.set_trace()

		# Create dictionary of time indeces
		time_dict = {}
		counter = 0
		for topic, msg, t in this_bag.read_messages(topics=[velo_topic]):
			if topic == velo_topic:
				time_dict[str(t)] = counter
				counter += 1


		# publish all the messages in order
		msg_count = 0
		for topic, msg, t in this_bag.read_messages(topics=[velo_topic, tf_topic, image_topic]):
			# if msg_count == 5:
			# 	msg_count = 0
			# 	# time.sleep(1.0)
			# 	_ = raw_input("press to continue...\n")
			# 	# pdb.set_trace()
			# 	# time.sleep(2.2)

			if topic == velo_topic:
				pub_pointcloud.publish(msg)
				callback_1(msg,this_pcl_path,str(msg_count))
				# lidar_msg_createpcl(msg,this_pcl_path,str(msg_count))
				# pub_clock.publish(t)
				# print(t)

			elif topic == tf_topic:
				pub_tf.publish(msg)
				# pub_clock.publish(t)
				# print(t)

			# elif topic == tfs_topic:
			# 	pub_tf_static.publish(msg)
			# 	pub_clock.publish(t)
			# 	# print(t)

			elif topic == image_topic:
				pub_image.publish(msg)
				# pub_clock.publish(t)
				# print(t)

			# elif topic == camera_info_topic:
			# 	pub_camera_info.publish(msg)
			# 	pub_clock.publish(t)
				# print(t)

			msg_count += 1
			
		this_bag.close()

		# rospy.spin()