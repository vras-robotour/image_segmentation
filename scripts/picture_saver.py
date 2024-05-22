#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


count = 0
snap = 0
pic = 0



def selector_callback(msg):
    global count
    global snap
    # rospy.loginfo(prdel)#rgb8; jpeg compressed brg8
    #rospy.loginfo(count)
    if(snap):
        # Convert uint8 list to bytes
        byte_data = bytes(msg.data)
        file_name = "tradr_20_5_24" + str(count) + ".jpg"
        # Write bytes to file
        with open(file_name, 'wb') as file:
            file.write(byte_data)
        rospy.loginfo("tradr_20_5_24" + str(count) + ".jpg")
        count += 1
        snap = False
    # rospy.loginfo('dylka:', len( msg.data)) #cca 64k
    # publisher = rospy.Publisher('/reactive_control/sector_dists', 
    #         SectorDistances, queue_size=50)
    # dst_msg = SectorDistances()
    #dst_msg.time_stamp = msg.time_increment
    # publisher.publish(dst_msg)


def activate_cb(msg: Bool):
    global snap
    snap = msg.data
    rospy.loginfo("state: " + str(snap))




def counter_callback(msg):
    global count
    global pic
    # rospy.loginfo(prdel)#rgb8; jpeg compressed brg8
    #rospy.loginfo(count)
    if(not (count%10)):
        
        # Convert uint8 list to bytes
        byte_data = bytes(msg.data)
        file_name = "spot_front_b" + str(pic) + ".jpg"
        # Write bytes to file
        with open(file_name, 'wb') as file:
            file.write(byte_data)
        rospy.loginfo("spot_front_b" + str(pic) + ".jpg")
        pic +=1

    count += 1



if __name__ == "__main__":
    rospy.init_node("my_subscriber")
    rospy.loginfo("Image saver start")
    #prdel = 0
    subscriber = rospy.Subscriber(
        '/viz/camera_4/image/compressed', CompressedImage, selector_callback)
    subscriber = rospy.Subscriber(
        '/pic_saver/activate', Bool, activate_cb)
    rospy.spin()




#camera topics
    #basic
    #'/camera/color/image_raw/compressed'

    #spot front
    #'/camera_front/image_color/compressed'


#rosbag play 'bag_file' -r 'speed_number'  --topics 'topic_1'
#rosbag play spot -r 10  --topics /camera_front/image_color/compressed
#rosbag play spot -r 10  --topics /camera/color/image_raw/compressed

#command for copying data from rci server
    #scp -r dasekfil@subtdata.felk.cvut.cz:/data/robotour2024/rosbags /home/fila/

#command for uploading to server
    #scp -r rugd_robotour.zip dasekfil@subtdata.felk.cvut.cz:/data/robotour2024/RUGD/


