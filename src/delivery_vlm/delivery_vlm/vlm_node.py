#!/usr/bin/env python3
"""vlm_node: 영수증 분석 / 벨 검증 서비스를 제공하는 노드.

제공 서비스:
- /analyze_receipt (delivery_interfaces/AnalyzeReceipt)
    영수증 이미지 → 목적지 + 벨 누름 여부
- /verify_bell (delivery_interfaces/VerifyBell)
    카메라 이미지 → 벨 여부 + 화면상 위치

ANTHROPIC_API_KEY 환경변수가 설정된 터미널에서 실행해야 한다.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import cv2
import numpy as np

from delivery_interfaces.srv import AnalyzeReceipt, VerifyBell
from delivery_vlm.bell_vlm import BellVLM


def imgmsg_to_bgr(msg):
    """sensor_msgs/Image → OpenCV BGR ndarray (cv_bridge 없이 수동 변환).

    NumPy 2.x + pip OpenCV 환경에서 cv_bridge ABI 충돌을 피하기 위함.
    """
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1
    )
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    # bgr8 등은 그대로 사용 (downstream cv2 연산을 위해 복사본 반환)
    return raw.copy()


class VlmNode(Node):
    def __init__(self):
        super().__init__('vlm_node')

        self.vlm = BellVLM(logger=self.get_logger())

        cb_group = ReentrantCallbackGroup()

        self.srv_receipt = self.create_service(
            AnalyzeReceipt, 'analyze_receipt',
            self.handle_analyze_receipt, callback_group=cb_group
        )
        self.srv_verify = self.create_service(
            VerifyBell, 'verify_bell',
            self.handle_verify_bell, callback_group=cb_group
        )

        self.get_logger().info('vlm_node 준비 완료. (/analyze_receipt, /verify_bell)')

    def handle_analyze_receipt(self, request, response):
        self.get_logger().info('[analyze_receipt] 영수증 분석 요청 수신')
        try:
            bgr = imgmsg_to_bgr(request.image)
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')
            response.success = False
            response.destination = ''
            response.press_bell = False
            return response

        destination, press_bell, ok = self.vlm.read_receipt(bgr)
        response.success = ok
        response.destination = destination
        response.press_bell = press_bell
        self.get_logger().info(
            f'[analyze_receipt] 결과: success={ok}, '
            f'destination="{destination}", press_bell={press_bell}'
        )
        return response

    def handle_verify_bell(self, request, response):
        try:
            bgr = imgmsg_to_bgr(request.image)
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')
            response.is_bell = False
            response.position = '없음'
            return response

        is_bell, position = self.vlm.verify_and_locate(bgr)
        response.is_bell = is_bell
        response.position = position
        self.get_logger().info(
            f'[verify_bell] 결과: is_bell={is_bell}, position={position}'
        )
        return response


def main():
    rclpy.init()
    node = VlmNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
