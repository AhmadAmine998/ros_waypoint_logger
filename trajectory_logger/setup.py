from setuptools import setup

package_name = 'trajectory_logger'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=[
        'setuptools',
        'pandas',
        'scipy',
        'numpy'
    ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Logs odometry and saves interpolated track trajectory to CSV.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_logger_node = trajectory_logger.trajectory_logger_node:main',
        ],
    },
)
