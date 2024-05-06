#!/usr/bin/env python
# this ROS node publish camera info of segmentation
#for point_cloud_color node
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo

def republish_callback(cam_info:CameraInfo):
    cam_info.height = 550
    cam_info.width = 688


def talker():
    pub = rospy.Publisher('/viz/camera_4/seg_camera_info', CameraInfo, queue_size=10)
    sub = rospy.Subscriber('/viz/camera_4/camera_info', CameraInfo, )
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('camera_info/segmentatoin', anonymous=True)