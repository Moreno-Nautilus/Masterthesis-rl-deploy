"""Small URDF-to-PyKDL parser used when ROS does not ship kdl_parser_py."""

from __future__ import annotations

from collections import defaultdict, deque
import xml.etree.ElementTree as ET


def _floats(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    parts = tuple(float(x) for x in value.split())
    if len(parts) != len(default):
        raise ValueError(f"Expected {len(default)} floats, got {len(parts)} in {value!r}")
    return parts


def _origin_frame(kdl, elem: ET.Element | None):
    if elem is None:
        return kdl.Frame.Identity()
    xyz = _floats(elem.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _floats(elem.get("rpy"), (0.0, 0.0, 0.0))
    return kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))


def _joint_axis(kdl, joint_elem: ET.Element, origin):
    axis_elem = joint_elem.find("axis")
    axis = _floats(axis_elem.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0))
    return origin.M * kdl.Vector(*axis)


def _kdl_joint(kdl, joint_elem: ET.Element, origin):
    name = joint_elem.get("name", "")
    joint_type = joint_elem.get("type", "fixed")
    if joint_type == "fixed":
        return kdl.Joint(name, kdl.Joint.Fixed)
    if joint_type in ("revolute", "continuous"):
        return kdl.Joint(name, origin.p, _joint_axis(kdl, joint_elem, origin), kdl.Joint.RotAxis)
    if joint_type == "prismatic":
        return kdl.Joint(name, origin.p, _joint_axis(kdl, joint_elem, origin), kdl.Joint.TransAxis)
    raise ValueError(f"Unsupported URDF joint type {joint_type!r} for joint {name!r}")


def tree_from_string(robot_description: str):
    """Parse a URDF string into ``(ok, PyKDL.Tree)``.

    This mirrors the small subset of ``kdl_parser_py.urdf.treeFromString`` needed by
    the deploy node: fixed, revolute/continuous, and prismatic joints.
    """

    import PyKDL as kdl

    try:
        root = ET.fromstring(robot_description)
    except ET.ParseError:
        return False, None
    links = {elem.get("name") for elem in root.findall("link") if elem.get("name")}
    if not links:
        return False, None

    joints = []
    child_links = set()
    children_by_parent = defaultdict(list)
    for elem in root.findall("joint"):
        parent_elem = elem.find("parent")
        child_elem = elem.find("child")
        parent = parent_elem.get("link") if parent_elem is not None else None
        child = child_elem.get("link") if child_elem is not None else None
        if not parent or not child:
            continue
        if parent not in links or child not in links:
            continue
        joints.append((parent, child, elem))
        child_links.add(child)
        children_by_parent[parent].append((child, elem))

    root_candidates = sorted(links - child_links)
    if len(root_candidates) != 1:
        return False, None

    tree = kdl.Tree(root_candidates[0])
    added_links = {root_candidates[0]}
    queue = deque([root_candidates[0]])
    added_joints = 0
    while queue:
        parent = queue.popleft()
        for child, joint_elem in children_by_parent[parent]:
            try:
                origin = _origin_frame(kdl, joint_elem.find("origin"))
                segment = kdl.Segment(child, _kdl_joint(kdl, joint_elem, origin), origin)
            except ValueError:
                return False, None
            if not tree.addSegment(segment, parent):
                return False, None
            added_links.add(child)
            added_joints += 1
            queue.append(child)

    if added_joints != len(joints) or added_links != links:
        return False, None
    return True, tree
