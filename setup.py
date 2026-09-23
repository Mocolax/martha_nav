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
            'map_publisher = martha_nav.ros.map_publisher:main',
            'ground_truth_tf = martha_nav.ros.ground_truth_tf:main',
            'planner_node = martha_nav.ros.planner_node:main',
            'policy_node = martha_nav.ros.policy_node:main',
            'cmd_vel_bridge = martha_nav.ros.cmd_vel_bridge:main',
        ],
    },
)
