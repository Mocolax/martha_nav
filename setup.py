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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mocolax',
    maintainer_email='nicolas58david@hotmail.com',
    description='PPO local planner for the Martha robot (thesis).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': []},
)
