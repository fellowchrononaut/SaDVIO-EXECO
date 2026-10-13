# ROS map interface (`ros_map_interface`)

What SaDVIO's ROS node (`vio_ros`) publishes for a map server such as MeshCSLAM's: the loop-corrected trajectory,
the anchored dense meshes and their corrected poses. Standard message types only, so no message package is needed.
Switched on by `ros_map_interface: 1`, the default. It is published even with `enable_visu: 0`, while the viewer
topics then stay off.

## Frames

- `world`: SaDVIO's odometry frame (the sliding window's), as before. `world -> robot` is published per frame
  (viewer, `enable_visu`).
- `map`: the loop-corrected frame. The node publishes **TF `map -> world`** for every frame. It is the loop closure's
  current correction of the odometry frame, the usual ROS `map -> odom` pattern.
  - It is the identity until the first loop, and it jumps when a loop is closed.
  - The robot's corrected pose is `map -> world -> robot`.
  - In mono VO the correction is a similarity. TF carries its rigid part only, so use `vo_pose_corrected` there.
- After a re-initialization the odometry restarts in a new world. `map -> world` is that segment's correction: the
  identity until a loop joins the segment to an earlier one.

## Topics

| topic | type | QoS | content |
|---|---|---|---|
| `/tf` (`map -> world`) | `tf2_msgs/TFMessage` | default | per frame: the current correction |
| `vo_pose_corrected` | `geometry_msgs/PoseStamped` | depth 100 | per frame: the frame's corrected pose in `map` (`LoopClosure::correct`, exact in mono too) |
| `keyframes_corrected` | `nav_msgs/Path` | reliable, transient local, depth 1 | every keyframe that left the window, at its corrected pose; each pose stamped with its keyframe. Re-sent after every new keyframe or pose graph solve (at most 5 Hz) |
| `dense_anchor_mesh` | `visualization_msgs/Marker` | reliable, depth 1000 | `dense_anchored` only: each closed anchor's geometry, sent once (see below) |
| `dense_anchor_poses` | `nav_msgs/Path` | reliable, transient local, depth 1 | `dense_anchored` only: every closed anchor's corrected pose, in anchor id order, stamped with the anchor keyframe. Re-sent at every map save (`dense_anchored_save_every` keyframes) |

`dense_anchor_mesh` details:
- `TRIANGLE_LIST`, or `POINTS` for `dense_mesh_method: none`.
- `ns` is `dense_anchor`, `id` is the anchor id, and the stamp is the anchor keyframe's timestamp.
- `points` are in the anchor's frame (its rectified left camera). `pose` is the anchor's corrected pose when the
  anchor closed, so the marker shows in RViz as is.
- A later correction does not re-send the mesh. The anchor's new pose arrives in `dense_anchor_poses` (same stamp):
  the consumer keeps the mesh and re-places it.

Anchors (`doc/dense_anchored_maps.md`):
- Per-keyframe methods (zncc, pd, per-keyframe gp, none) have one anchor per keyframe.
- Fused methods (gp global map, vdbgpdf) have one per submap of `dense_submap_kfs` keyframes. A submap closes when it
  is full or at a re-initialization. When a loop moves the window, the open submap is re-expressed in the moved world
  and continues.

With `dense_anchored` and the interface on, the viewer's `dense_mesh` (the global or fused mesh) is in `map`. Loop closure
markers (`loop_traj`, `loop_edges`, `loop_new`) are in `map` when the interface is on, and in `world` otherwise.

## For a map server

1. Subscribe to `dense_anchor_mesh` before the run starts: meshes are sent once. The same geometry is also on disk
   (`log_slam/dense_anchors/anchor_NNNNN.ply`, anchor frame, and `anchors.csv` with the odometry and corrected poses).
2. Keep each mesh by its stamp (or id), and place it at the latest pose with the same stamp in `dense_anchor_poses`.
3. Use `keyframes_corrected` for the trajectory and `vo_pose_corrected` or TF for the live pose.

## Checking it

`doc/vio_imu_fix/tools/run_sadvio_ros.py --record-interface` runs `record_map_interface.py` next to the node during a
bag replay and writes `interface.json`:
- the messages per topic;
- how often `map -> world` changed;
- the keyframes in the last path;
- whether every anchor mesh arrived once;
- how many anchor poses match a mesh, and how far the anchors moved after their mesh was sent.
