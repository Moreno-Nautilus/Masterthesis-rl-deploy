#!/usr/bin/env python3
"""Publish static BOX collision objects (table, cameras, fixtures) to move_group's planning scene.

So MoveIt (and the preinsert planner) plans AROUND the table and static cameras instead of sweeping
through them. Same mechanism as the supervisor's publish_camera_scene_objects.py -- CollisionObjects
on the planning_scene topic as PlanningScene(is_diff=True), republished periodically because the topic
isn't latched -- but namespace-correct for our move_group and using simple boxes (no meshes/extrinsics).

Run it in its own terminal (no build needed):
    source /opt/ros/humble/setup.bash
    source ~/kuka_fri_omar_ws/install/setup.bash      # for moveit_msgs
    python3 scene_collision_publisher.py --config scene_collisions.yaml

Verify in RViz before trusting it: relaunch move_group with rviz:=true, add a PlanningScene display on
/lbr_dual_arm_y_gripper/monitored_planning_scene -- the boxes should appear where the real table/camera
are, and NOT overlap the robot (an overlapping box makes every plan fail "start state in collision").

--remove clears the objects again.
"""
from __future__ import annotations

import argparse

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneWorld
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header

# Sensible starting scene if no --config is given: a table covering the work area IN FRONT of the
# robot (x from ~0.3..1.2), top just below the socket (z=-0.01), NOT enclosing the arm bases at x~0.
# VERIFY against your real cell + in RViz; wrong geometry either fails to protect or blocks valid plans.
DEFAULT_BOXES = [
    {"id": "work_table", "size": [0.9, 1.4, 0.10], "pose": [0.75, 0.0, -0.06]},
]


def _make_box(box: dict, frame_id: str, stamp) -> CollisionObject:
    obj = CollisionObject()
    obj.header = Header(frame_id=frame_id, stamp=stamp)
    obj.id = str(box["id"])
    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [float(v) for v in box["size"]]  # [x, y, z] full extents, meters
    p = box["pose"]
    q = box.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])
    pose = Pose(
        position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
        orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
    )
    obj.primitives = [prim]
    obj.primitive_poses = [pose]
    obj.operation = CollisionObject.ADD
    return obj


class SceneCollisionPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("scene_collision_publisher")
        self._pub = self.create_publisher(PlanningScene, args.topic, 1)
        self._frame = args.frame
        self._remove = args.remove

        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            self._boxes = cfg.get("boxes", [])
            self._frame = cfg.get("frame_id", self._frame)
        else:
            self._boxes = DEFAULT_BOXES

        ids = [b["id"] for b in self._boxes]
        self.get_logger().info(
            f"{'REMOVING' if self._remove else 'Publishing'} {len(self._boxes)} collision boxes "
            f"{ids} in frame '{self._frame}' -> {args.topic}"
        )
        self.create_timer(args.republish_period_s, self._publish)
        self._publish()

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        objs = []
        for box in self._boxes:
            obj = _make_box(box, self._frame, stamp)
            if self._remove:
                obj.operation = CollisionObject.REMOVE
            objs.append(obj)
        self._pub.publish(
            PlanningScene(is_diff=True, world=PlanningSceneWorld(collision_objects=objs))
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish table/camera collision boxes to move_group.")
    parser.add_argument("--topic", default="/lbr_dual_arm_y_gripper/planning_scene")
    parser.add_argument("--frame", default="base_link")
    parser.add_argument("--config", default=None, help="YAML with a 'boxes:' list (see scene_collisions.yaml).")
    parser.add_argument("--republish-period-s", type=float, default=2.0)
    parser.add_argument("--remove", action="store_true", help="Remove the objects instead of adding them.")
    args, _ = parser.parse_known_args()  # tolerate --ros-args

    rclpy.init()
    node = SceneCollisionPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
