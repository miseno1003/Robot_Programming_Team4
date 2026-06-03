#!/usr/bin/env python3
"""VLM 정렬 단독 테스트 — Nav2 없이, TwistStamped, 벨 찾을 때까지 무한 탐색"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
import time
import anthropic
import base64
import cv2
import numpy as np


class AlignTest(Node):
    def __init__(self):
        super().__init__('align_test')
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.camera_msg = None
        self.cam_sub = self.create_subscription(
            Image, '/camera/image_raw', self._cam_cb, 1)
        self.vlm_client = anthropic.Anthropic()

    def _cam_cb(self, msg):
        self.camera_msg = msg

    def _pub_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def _stop(self):
        self._pub_cmd(0.0, 0.0)

    def _get_image_b64(self):
        self.camera_msg = None
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.camera_msg is not None:
                break
        if self.camera_msg is None:
            return None

        msg = self.camera_msg
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        else:
            bgr = raw
        _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(jpeg.tobytes()).decode('utf-8')

    def _ask_vlm(self, b64, question):
        response = self.vlm_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": question},
                ],
            }],
        )
        return response.content[0].text

    def _rotate(self, angular_z, duration):
        end_time = time.time() + duration
        while time.time() < end_time:
            self._pub_cmd(0.0, angular_z)
            time.sleep(0.05)
        self._stop()
        time.sleep(0.3)

    def _move_forward(self, distance, speed=0.05):
        self.get_logger().info(f'벨을 향해 {distance*100:.0f}cm 전진')
        end_time = time.time() + (distance / speed)
        while time.time() < end_time:
            self._pub_cmd(speed, 0.0)
            time.sleep(0.05)
        self._stop()
        time.sleep(0.5)
        self.get_logger().info('전진 완료!')

    def run(self):
        self.get_logger().info('=== VLM 정렬 테스트 시작 (Ctrl+C로 중단) ===')
        attempt = 0

        while True:
            attempt += 1
            self.get_logger().info(f'--- 시도 {attempt} ---')

            b64 = self._get_image_b64()
            if b64 is None:
                self.get_logger().error('카메라 이미지 수신 실패, 재시도...')
                time.sleep(1.0)
                continue

            answer = self._ask_vlm(
                b64,
                "이 이미지에 빨간색 벨(둥근 버튼)이 보이나요? "
                "반드시 다음 형식으로만 답하세요:\n"
                "1) 안 보임\n"
                "2) 보임-왼쪽\n"
                "3) 보임-중앙\n"
                "4) 보임-오른쪽\n"
                "번호와 해당 텍스트만 답하세요."
            )
            self.get_logger().info(f'VLM 응답: {answer}')

            if '중앙' in answer or '3)' in answer:
                self.get_logger().info('벨이 화면 중앙! 정렬 완료!')
                self._move_forward(0.10)

                # 최종 VLM 확인
                b64 = self._get_image_b64()
                if b64:
                    final = self._ask_vlm(
                        b64,
                        "빨간색 벨이 가까이 보이나요? "
                        "간단히 답해주세요."
                    )
                    print(f"\n=== VLM 최종 확인 ===")
                    print(final)

                print(f"\n=== 완료! (총 {attempt}회 시도) ===")
                return True

            elif '왼쪽' in answer or '2)' in answer:
                self.get_logger().info('왼쪽 → 반시계 회전')
                self._rotate(0.3, 0.5)
            elif '오른쪽' in answer or '4)' in answer:
                self.get_logger().info('오른쪽 → 시계 회전')
                self._rotate(-0.3, 0.5)
            else:
                self.get_logger().info('안 보임 → 반시계 탐색')
                self._rotate(0.4, 0.8)

            time.sleep(0.5)


def main():
    rclpy.init()
    node = AlignTest()

    try:
        node.run()
    except KeyboardInterrupt:
        node._stop()
        print('\n긴급 정지!')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
