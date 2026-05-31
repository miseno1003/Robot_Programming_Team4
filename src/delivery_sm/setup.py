import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'delivery_sm'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kjs',
    maintainer_email='kjs@todo.todo',
    description='상태기계 메인 노드 + 전체 실행 런치',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_machine_node = delivery_sm.state_machine_node:main',
        ],
    },
)
