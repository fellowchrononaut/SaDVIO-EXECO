#!/usr/bin/env python3
"""Convert an EuRoC-format sequence (TUM-VI mav0: cam0, cam1, imu0) into a ROS 2 Humble bag (sqlite3).

Runs inside the sad_vio_dense container with its ROS 2 Python (rosbag2_py, rclpy serialization) and PIL, so the bag
is written by the same rosbag2 version that plays it. Images keep their native encoding (TUM-VI 512_16: mono16, as in
the official TUM-VI bags); the ROS node converts them to mono8 with cv_bridge. Message and record stamps are the
dataset timestamps, so `ros2 bag play` reproduces the original timing.
Topics: /cam0/image_raw, /cam1/image_raw (sensor_msgs/Image), /imu0 (sensor_msgs/Imu).

Usage (from the host):
  docker exec -i sad_vio_dense bash -c 'source /opt/ros/humble/setup.bash && python3 - <mav0> <out_bag>' \
      < euroc_to_ros2bag.py
"""
import array, csv, sys
from pathlib import Path

import numpy as np
import rosbag2_py
from PIL import Image as PILImage
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, Imu

mav0, out = Path(sys.argv[1]), sys.argv[2]


def stamp(msg, ns):
    msg.header.stamp.sec, msg.header.stamp.nanosec = ns // 10**9, ns % 10**9


def rows(path):
    with open(path) as f:
        return [r for r in csv.reader(f) if r and not r[0].startswith('#')]


writer = rosbag2_py.SequentialWriter()
writer.open(rosbag2_py.StorageOptions(uri=out, storage_id='sqlite3'),
            rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
for name, typ in [('/cam0/image_raw', 'sensor_msgs/msg/Image'), ('/cam1/image_raw', 'sensor_msgs/msg/Image'),
                  ('/imu0', 'sensor_msgs/msg/Imu')]:
    writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format='cdr'))

msgs = []  # (t_ns, kind, payload)
for cam in ('cam0', 'cam1'):
    msgs += [(int(r[0]), cam, r[1].strip()) for r in rows(mav0 / cam / 'data.csv')]
msgs += [(int(r[0]), 'imu', [float(x) for x in r[1:7]]) for r in rows(mav0 / 'imu0' / 'data.csv')]
msgs.sort(key=lambda m: (m[0], m[1] != 'imu'))  # IMU first on equal stamps

n = {'cam0': 0, 'cam1': 0, 'imu': 0}
for t, kind, payload in msgs:
    if kind == 'imu':
        m = Imu()
        stamp(m, t)
        m.header.frame_id = 'imu0'
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = payload[:3]
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = payload[3:]
        m.orientation_covariance[0] = -1.0  # no orientation
        writer.write('/imu0', serialize_message(m), t)
    else:
        a = np.asarray(PILImage.open(mav0 / kind / 'data' / payload))
        m = Image()
        stamp(m, t)
        m.header.frame_id = kind
        m.height, m.width = a.shape[:2]
        if a.dtype == np.uint8:
            m.encoding, m.step = 'mono8', a.shape[1]
        else:
            a = a.astype('<u2')
            m.encoding, m.step = 'mono16', 2 * a.shape[1]
        m.is_bigendian = 0
        m.data = array.array('B', a.tobytes())
        writer.write(f'/{kind}/image_raw', serialize_message(m), t)
    n[kind] += 1
print(f'{out}: {n["cam0"]} + {n["cam1"]} images, {n["imu"]} IMU samples')
