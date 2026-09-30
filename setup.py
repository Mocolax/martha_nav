from glob import glob

from setuptools import find_packages, setup

package_name = 'martha_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mocolax',
    maintainer_email='nicolas58david@hotmail.com',
    description='PPO local planner for the Martha robot (thesis).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_policy = martha_nav.learning.train:main',
            'evaluate_2d = martha_nav.learning.evaluate:main',
            'world_map_publisher = martha_nav.ros.world_map_publisher:main',
            'gazebo_ground_truth_tf = martha_nav.ros.gazebo_ground_truth_tf:main',
            'global_planner = martha_nav.ros.global_planner:main',
            'ppo_local_planner = martha_nav.ros.ppo_local_planner:main',
            'mecanum_cmd_vel_bridge = martha_nav.ros.mecanum_cmd_vel_bridge:main',
            'evaluate_gazebo = martha_nav.ros.evaluate_gazebo:main',
            'esp32_bridge = martha_nav.ros.esp32_bridge:main',
        ],
    },
)
