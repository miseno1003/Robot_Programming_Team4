#!/usr/bin/env python3
"""
button_monitor.py

벨 버튼의 contact sensor 토픽을 구독해서
'눌렸다/안 눌렸다'를 단순한 Bool 토픽으로 변환해 발행한다.

구독: /world/bell_world/.../contact   (ros_gz_interfaces/Contacts)
발행: /button_pressed                   (std_msgs/Bool)
       /button_press_count              (std_msgs/Int32, 디버그용)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32
from ros_gz_interfaces.msg import Contacts


CONTACT_TOPIC = '/red_button/contact'

# 막대기 collision 이름의 부분 문자열
# Gazebo 출력 확인 결과: "burger_with_pole::pole_link::pole_collision"
STICK_NAME_KEYWORD = 'pole'


class ButtonMonitor(Node):
    def __init__(self):
        super().__init__('button_monitor')

        self.pressed = False
        self.press_count = 0
        self.was_pressed_last_frame = False

        # 발행자
        self.pub_pressed = self.create_publisher(Bool, '/button_pressed', 10)
        self.pub_count = self.create_publisher(Int32, '/button_press_count', 10)

        # 구독자
        self.create_subscription(
            Contacts, CONTACT_TOPIC, self.contact_cb, 10
        )

        # 주기적으로 현재 상태 publish (10Hz)
        self.create_timer(0.1, self.publish_state)

        self.get_logger().info(
            f'Button monitor started.\n'
            f'  Subscribing: {CONTACT_TOPIC}\n'
            f'  Keyword:     "{STICK_NAME_KEYWORD}"'
        )

    def contact_cb(self, msg: Contacts):
        """Gazebo contact 메시지를 받아 막대기가 닿았는지 판정"""
        is_pressed_now = False

        for contact in msg.contacts:
            name1 = contact.collision1.name.lower()
            name2 = contact.collision2.name.lower()
            if STICK_NAME_KEYWORD in name1 or STICK_NAME_KEYWORD in name2:
                is_pressed_now = True
                break

        self.pressed = is_pressed_now

        # rising edge에서만 카운트 (디바운스)
        if is_pressed_now and not self.was_pressed_last_frame:
            self.press_count += 1
            self.get_logger().info(
                f'🔔 Button pressed! (count={self.press_count})'
            )

        self.was_pressed_last_frame = is_pressed_now

    def publish_state(self):
        self.pub_pressed.publish(Bool(data=self.pressed))
        self.pub_count.publish(Int32(data=self.press_count))


def main():
    rclpy.init()
    node = ButtonMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()