#!/usr/bin/env python3
"""Record SaDVIO's ROS map interface (ros_map_interface, doc/loop_closure/ros_map_interface.md) during a run and check
it: counts per topic, the map -> world transform (does it change after loops?), the corrected keyframes, the anchor
meshes (each id once?) and the anchor poses (do they cover the meshes, and how far have the anchors moved since their
mesh was sent?). Runs in the container (rclpy), on the node's ROS domain. Writes its JSON summary every second and on
SIGINT / SIGTERM (rclpy's own signal handling is off, it would swallow them).

usage: record_map_interface.py <out.json>
"""
import json
import math
import os
import signal
import sys

import rclpy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker


def stamp_ns(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def pos(p):
    return (p.position.x, p.position.y, p.position.z)


class Recorder(Node):
    def __init__(self):
        super().__init__('map_interface_recorder')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.n = {'vo_pose_corrected': 0, 'keyframes_corrected': 0, 'dense_anchor_mesh': 0, 'dense_anchor_poses': 0,
                  'tf_map_world': 0}
        self.tf_moves = 0          # map -> world updates that differ from the previous one by > 1 mm
        self.tf_last = None
        self.tf_max_offset = 0.0   # largest |translation| of map -> world
        self.kf_last = []
        self.mesh_ids, self.mesh_pose, self.mesh_faces = [], {}, 0
        self.poses_last = []
        self.create_subscription(PoseStamped, 'vo_pose_corrected', self.on_pose, 100)
        self.create_subscription(Path, 'keyframes_corrected', self.on_kfs, latched)
        self.create_subscription(Path, 'dense_anchor_poses', self.on_anchor_poses, latched)
        self.create_subscription(Marker, 'dense_anchor_mesh', self.on_mesh, QoSProfile(depth=1000,
                                 reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(TFMessage, 'tf', self.on_tf, 100)

    def on_pose(self, m):
        self.n['vo_pose_corrected'] += 1

    def on_kfs(self, m):
        self.n['keyframes_corrected'] += 1
        self.kf_last = [(stamp_ns(p.header), pos(p.pose)) for p in m.poses]

    def on_anchor_poses(self, m):
        self.n['dense_anchor_poses'] += 1
        self.poses_last = [(stamp_ns(p.header), pos(p.pose)) for p in m.poses]

    def on_mesh(self, m):
        self.n['dense_anchor_mesh'] += 1
        self.mesh_ids.append(m.id)
        self.mesh_pose[stamp_ns(m.header)] = pos(m.pose)
        self.mesh_faces += len(m.points) // 3

    def on_tf(self, m):
        for t in m.transforms:
            if t.header.frame_id == 'map' and t.child_frame_id == 'world':
                self.n['tf_map_world'] += 1
                v = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z,
                     t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
                if self.tf_last is not None and max(abs(a - b) for a, b in zip(v, self.tf_last)) > 1e-3:
                    self.tf_moves += 1
                self.tf_last = v
                self.tf_max_offset = max(self.tf_max_offset, math.sqrt(sum(x * x for x in v[:3])))

    def summary(self):
        moved = [math.dist(p, self.mesh_pose[ts]) for ts, p in self.poses_last if ts in self.mesh_pose]
        return dict(counts=self.n, tf_map_world_changes=self.tf_moves, tf_map_world_max_offset_m=self.tf_max_offset,
                    keyframes_in_last_path=len(self.kf_last), anchor_meshes=len(self.mesh_ids),
                    anchor_mesh_ids_unique=len(set(self.mesh_ids)) == len(self.mesh_ids),
                    anchor_mesh_triangles=self.mesh_faces, anchors_in_last_poses=len(self.poses_last),
                    anchor_poses_matching_a_mesh=len(moved),
                    anchor_moved_since_mesh_m=dict(max=max(moved, default=0.0),
                                                   mean=sum(moved) / len(moved) if moved else 0.0))


def main():
    out = sys.argv[1]
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    rec = Recorder()

    def dump():
        with open(out + '.tmp', 'w') as f:
            json.dump(rec.summary(), f, indent=1)
        os.replace(out + '.tmp', out)

    def stop(*_):
        dump()
        os._exit(0)

    rec.create_timer(1.0, dump)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    rclpy.spin(rec)


if __name__ == '__main__':
    main()
