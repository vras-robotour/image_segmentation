#!/usr/bin/env python
import rospy
import struct
import numpy as np
from ros_numpy import msgify, numpify
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

def process_point_cloud(point_cloud_msg: PointCloud2):
    if msg_delay(point_cloud_msg) > 0.5:
        rospy.logwarn("Message delay is too high")
        return

    # Convert PointCloud2 to numpy array
    pc_array = numpify(point_cloud_msg)

    # Extract fields
    x = pc_array['x']
    y = pc_array['y']
    z = pc_array['z']
    rgb_float = pc_array['rgb'].astype(np.uint32)

    # Unpack RGB values
    r = (rgb_float >> 16) & 0xFF
    g = (rgb_float >> 8) & 0xFF
    b = rgb_float & 0xFF

    # Compute cost from the intensity of the red channel
    # If the r == 0 that means that there is no prediction (out of camera FOV)
    # Other values should be 1-255, therefore we have normalize them
    cost = np.where(r == 0, 0.5, r / 255.0)

    # Create a structured array with the processed points
    processed_points = np.rec.fromarrays([x, y, z, r, g, b, cost],
                                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                                                ('cost', 'f4')])

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = point_cloud_msg.header.frame_id

    # Convert numpy array back to PointCloud2
    filtered_cloud_msg = msgify(PointCloud2, processed_points)
    filtered_cloud_msg.header = header

    return filtered_cloud_msg


def msg_delay(msg: PointCloud2) -> float:
    return (rospy.Time.now() - msg.header.stamp).to_sec()

def point_cloud_callback(msg: PointCloud2):
    filtered_cloud_msg = process_point_cloud(msg)
    pub.publish(filtered_cloud_msg)


if __name__ == '__main__':
    rospy.init_node('semantic_traversability_node', log_level=rospy.DEBUG)

    rospy.Subscriber("cloud_in", PointCloud2, point_cloud_callback)

    pub = rospy.Publisher("cloud_out", PointCloud2, queue_size=1)

    rospy.spin()
