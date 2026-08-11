from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_deploy_inference.kinematics import KdlConfig, KdlKinematics  # noqa: E402


def _seven_joint_urdf() -> str:
    links = "\n".join(f'<link name="link_{i}"/>' for i in range(8))
    joints = "\n".join(
        f"""
        <joint name="joint_{i}" type="revolute">
          <parent link="link_{i - 1}"/>
          <child link="link_{i}"/>
          <origin xyz="0.1 0 0" rpy="0 0 0"/>
          <axis xyz="0 0 1"/>
          <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
        </joint>
        """
        for i in range(1, 8)
    )
    return f'<robot name="test_arm">{links}{joints}</robot>'


def test_kdl_kinematics_uses_local_parser_when_kdl_parser_py_is_missing() -> None:
    kin = KdlKinematics(
        KdlConfig(
            robot_description=_seven_joint_urdf(),
            base_link="link_0",
            tip_link="link_7",
            joint_names=tuple(f"joint_{i}" for i in range(1, 8)),
        )
    )

    q = np.zeros(7)
    pos, rot = kin.fk(q)
    jac = kin.jacobian(q)

    np.testing.assert_allclose(pos, [0.7, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)
    assert jac.shape == (6, 7)
    assert np.all(np.isfinite(jac))
