#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header


def process_point_cloud(point_cloud_msg: PointCloud2):
    rospy.logdebug(f"Fields: {point_cloud_msg.fields}")
    # Extract points from the incoming point cloud message
    points = point_cloud2.read_points(point_cloud_msg, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True)

    processed_points = []
    for point in points:
        x, y, z, r, g, b = point
        if r == 0:
            cost = 0.5
        else:
            cost = r / 255.0
        processed_points.append([x, y, z, r, g, b, cost])

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = point_cloud_msg.header.frame_id

    # Define the fields for the new PointCloud2 message
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="r", offset=12, datatype=PointField.UINT8, count=1),
        PointField(name="g", offset=13, datatype=PointField.UINT8, count=1),
        PointField(name="b", offset=14, datatype=PointField.UINT8, count=1),
        PointField(name="cost", offset=16, datatype=PointField.FLOAT32, count=1)
    ]

    # Create a new PointCloud2 message
    filtered_cloud_msg = point_cloud2.create_cloud(header, fields, processed_points)

    return filtered_cloud_msg

def point_cloud_callback(msg: PointCloud2):
    filtered_cloud_msg = process_point_cloud(msg)
    pub.publish(filtered_cloud_msg)

if __name__ == '__main__':
    rospy.init_node('semantic_traversability_node', log_level=rospy.DEBUG)

    rospy.Subscriber("cloud_in", PointCloud2, point_cloud_callback)

    pub = rospy.Publisher("cloud_out", PointCloud2, queue_size=10)

    rospy.spin()
