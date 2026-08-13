from setuptools import find_packages, setup

package_name = "rl_deploy_inference"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            ["config/deploy.yaml", "config/preinsert.yaml", "config/hole_align.yaml"],
        ),
        (
            f"share/{package_name}/launch",
            [
                "launch/deploy_inference.launch.py",
                "launch/preinsert_planner.launch.py",
                "launch/hole_align_planner.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="moreno",
    maintainer_email="moreno@example.com",
    description="Real-robot inference node for the iiwa7 cooling-screw insertion RL policy.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inference_node = rl_deploy_inference.inference_node:main",
            "obs_parity = rl_deploy_inference.parity_check:main",
            "preinsert_planner = rl_deploy_inference.preinsert_planner:main",
            "hole_align_planner = rl_deploy_inference.hole_align_planner:main",
        ],
    },
)
