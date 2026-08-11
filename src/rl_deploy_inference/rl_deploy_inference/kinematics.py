"""Kinematics adapters for real-time DLS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ik import rotmat_from_quat_wxyz


class KinematicsUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class KdlConfig:
    robot_description: str
    base_link: str
    tip_link: str
    joint_names: tuple[str, ...]
    fallback_tip_link: str = ""
    fallback_tip_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fallback_tip_offset_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


class KdlKinematics:
    """KDL FK/Jacobian wrapper with a safe link_ee + fixed TCP-offset fallback."""

    def __init__(self, cfg: KdlConfig):
        self.cfg = cfg
        try:
            import PyKDL as kdl
        except ImportError as exc:
            raise KinematicsUnavailable(
                "PyKDL is not importable. Install ROS KDL Python bindings before enabling motion."
            ) from exc

        try:
            from kdl_parser_py.urdf import treeFromString
        except ImportError:
            from .kdl_parser import tree_from_string as treeFromString

        ok, tree = treeFromString(cfg.robot_description)
        if not ok:
            raise KinematicsUnavailable("Could not parse robot_description into a KDL tree.")

        self._kdl = kdl
        self._using_fallback = False
        tip_link = cfg.tip_link
        try:
            self._chain = tree.getChain(cfg.base_link, tip_link)
        except Exception:
            self._chain = None
        if (self._chain is None or self._chain.getNrOfJoints() < 7) and cfg.fallback_tip_link:
            tip_link = cfg.fallback_tip_link
            self._chain = tree.getChain(cfg.base_link, tip_link)
            self._using_fallback = True

        if self._chain is None or self._chain.getNrOfJoints() < 7:
            raise KinematicsUnavailable(
                f"KDL chain {cfg.base_link}->{tip_link} has only "
                f"{0 if self._chain is None else self._chain.getNrOfJoints()} joints."
            )

        self._fk = kdl.ChainFkSolverPos_recursive(self._chain)
        self._jac = kdl.ChainJntToJacSolver(self._chain)
        self._chain_joint_names = self._joint_names_in_chain()
        self._joint_index = {name: i for i, name in enumerate(cfg.joint_names)}
        missing = [name for name in self._chain_joint_names if name not in self._joint_index]
        if missing:
            raise KinematicsUnavailable(
                "KDL chain contains joints not present in measured FRI order: " + ", ".join(missing)
            )
        self._offset = np.asarray(cfg.fallback_tip_offset_xyz, dtype=np.float64).reshape(3)
        self._offset_R = rotmat_from_quat_wxyz(np.asarray(cfg.fallback_tip_offset_quat_wxyz, dtype=np.float64))

    def _joint_names_in_chain(self) -> list[str]:
        # Only the MOVABLE joints consume a measured q; fixed joints (gripper mount/tcp frames) must be
        # excluded. Compare the joint-type ENUM against Fixed rather than the type-NAME string, because
        # PyKDL reports the fixed type as "Fixed" in some builds and "None" in others (the string check
        # silently let the gripper's fixed joints through -> false "not in FRI order" errors).
        fixed = int(self._kdl.Joint.Fixed)
        out: list[str] = []
        for i in range(self._chain.getNrOfSegments()):
            joint = self._chain.getSegment(i).getJoint()
            if int(joint.getType()) != fixed:
                out.append(joint.getName())
        return out

    def _jnt_array(self, q: np.ndarray):
        q = np.asarray(q, dtype=np.float64).reshape(len(self.cfg.joint_names))
        arr = self._kdl.JntArray(len(self._chain_joint_names))
        for chain_i, name in enumerate(self._chain_joint_names):
            arr[chain_i] = float(q[self._joint_index[name]])
        return arr

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame = self._kdl.Frame()
        status = self._fk.JntToCart(self._jnt_array(q), frame)
        if status < 0:
            raise KinematicsUnavailable(f"KDL FK failed with status {status}.")
        pos = np.array([frame.p[0], frame.p[1], frame.p[2]], dtype=np.float64)
        R = np.array([[frame.M[i, j] for j in range(3)] for i in range(3)], dtype=np.float64)
        if self._using_fallback:
            pos = pos + R @ self._offset
            R = R @ self._offset_R
        return pos, R

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        jac = self._kdl.Jacobian(len(self._chain_joint_names))
        status = self._jac.JntToJac(self._jnt_array(q), jac)
        if status < 0:
            raise KinematicsUnavailable(f"KDL Jacobian failed with status {status}.")
        mat_chain = np.array(
            [[jac[row, col] for col in range(jac.columns())] for row in range(6)],
            dtype=np.float64,
        )
        mat = np.zeros((6, len(self.cfg.joint_names)), dtype=np.float64)
        for chain_col, name in enumerate(self._chain_joint_names):
            mat[:, self._joint_index[name]] = mat_chain[:, chain_col]
        if self._using_fallback and np.linalg.norm(self._offset) > 0.0:
            _, R = self.fk_without_offset(q)
            r_base = R @ self._offset
            mat[0:3, :] = mat[0:3, :] + np.cross(mat[3:6, :].T, r_base).T
        return mat

    def fk_without_offset(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        was_using_fallback = self._using_fallback
        self._using_fallback = False
        try:
            return self.fk(q)
        finally:
            self._using_fallback = was_using_fallback
