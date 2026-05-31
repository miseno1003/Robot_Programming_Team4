from setuptools import find_packages, setup

package_name = 'delivery_vlm'

setup(
    name=package_name,
    version='0.0.1',
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
    description='VLM 인식 노드 (영수증 분석 / 벨 검증 서비스)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_node = delivery_vlm.vlm_node:main',
        ],
    },
)
