from setuptools import find_packages, setup

package_name = 'delivery_nav'

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
    description='네비게이션 + 벨 접근/누르기 + 모터 제어 노드',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nav_node = delivery_nav.nav_node:main',
        ],
    },
)
