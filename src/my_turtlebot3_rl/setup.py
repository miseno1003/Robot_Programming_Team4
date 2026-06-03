from setuptools import find_packages, setup

package_name = 'my_turtlebot3_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kjs',
    maintainer_email='kjs@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'button_monitor = my_turtlebot3_rl.button_monitor:main',
            'world_resetter = my_turtlebot3_rl.world_resetter:main',
            'bell_approach = my_turtlebot3_rl.bell_approach:main',
            'front_bell = my_turtlebot3_rl.front_bell:main',
        ],
    },
)
