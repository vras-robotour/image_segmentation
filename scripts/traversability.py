#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
from std_msgs.msg import Header


def process_point_cloud(point_cloud_msg):
    rospy.logdebug(f"Received point cloud message with {point_cloud_msg.width} points")
    # Print the point cloud message fields
    rospy.logdebug(f"Fields: {point_cloud_msg.fields}")
    # Example processing: filter points within a certain distance from the origin
    points = read_points(point_cloud_msg)
    filtered_points = []
    for p in points:
        if np.linalg.norm(p[0]) <= 1.0:  # Filter points within 1-meter radius
            filtered_points.append(p)

    # Convert filtered points back to PointCloud2 message
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = point_cloud_msg.header.frame_id

    return filtered_points, header


def point_cloud_callback(msg):
    filtered_points, header = process_point_cloud(msg)
    pub.publish(filtered_points, header)


if __name__ == '__main__':
    rospy.init_node('semantic_traversability_node', log_level=rospy.DEBUG)

    rospy.Subscriber("cloud_in", PointCloud2, point_cloud_callback)

    pub = rospy.Publisher("cloud_out", PointCloud2, queue_size=10)

    rospy.spin()
