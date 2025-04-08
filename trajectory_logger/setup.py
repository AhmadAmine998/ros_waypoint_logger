from setuptools import find_packages, setup
from glob import glob

package_name = 'trajectory_logger'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', ['config/params.yaml']),
    ],
    install_requires=[
        'setuptools',
        'pandas',
        'scipy',
        'numpy'
    ],
    zip_safe=True,
    maintainer='ahmad',
    maintainer_email='ahmadamine998@gmail.com',
    description='Logs odometry and saves interpolated track trajectory to CSV.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_logger_node = trajectory_logger.trajectory_logger_node:main',
        ],
    },
)