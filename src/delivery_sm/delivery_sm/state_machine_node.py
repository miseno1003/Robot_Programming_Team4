#!/usr/bin/env python3
"""state_machine_node: 전체 미션을 조율하는 메인 노드.

흐름:
1) 카메라로 영수증 캡처 → /analyze_receipt 서비스 호출 → (목적지, 벨여부)
2) /approach_bell 액션 전송 → 이동 + 벨 시퀀스 (or 회전) + 후진/정지
3) F키 대기 창 → F 누르면 /approach_bell(destination="HOME") 전송 → 홈 복귀
4) 루프. Q/ESC 로 종료.

사용 인터페이스(클라이언트):
- 서비스 /analyze_receipt → vlm_node
- 액션 /approach_bell → nav_node
토픽 구독:
- /camera/image_raw (미리보기/캡처)
- /delivery_status (진행상황 로그)
"""

import time
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Image
from std_msgs.msg import String

from delivery_interfaces.srv import AnalyzeReceipt
from delivery_interfaces.action import ApproachBell

from delivery_sm.receipt_ui import capture_receipt_and_decide, wait_for_f_key


def imgmsg_to_bgr(msg):
    """sensor_msgs/Image → OpenCV BGR ndarray (cv_bridge 없이 수동 변환)."""
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1
    )
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    return raw.copy()


class StateMachineNode(Node):
    def __init__(self):
        super().__init__('state_machine_node')

        self.camera_msg = None

        self._cb = ReentrantCallbackGroup()

        self.cam_sub = self.create_subscription(
            Image, '/camera/image_raw', self._cam_cb, 1,
            callback_group=self._cb
        )
        self.status_sub = self.create_subscription(
            String, '/delivery_status', self._status_cb, 10,
            callback_group=self._cb
        )

        self.analyze_client = self.create_client(
            AnalyzeReceipt, 'analyze_receipt', callback_group=self._cb
        )
        self.approach_client = ActionClient(
            self, ApproachBell, 'approach_bell', callback_group=self._cb
        )

        self.get_logger().info('state_machine_node 준비 완료.')

    def _cam_cb(self, msg):
        self.camera_msg = msg

    def _status_cb(self, msg):
        self.get_logger().info(f'[nav 상태] {msg.data}')

    # ----- UI 콜백용 -----
    def get_frame(self):
        if self.camera_msg is None:
            return None
        try:
            return imgmsg_to_bgr(self.camera_msg)
        except Exception:
            return None

    def analyze_current_frame(self):
        """현재 카메라 이미지를 /analyze_receipt 로 전송.
        반환: (success, destination, press_bell)"""
        if self.camera_msg is None:
            return False, '', False
        if not self.analyze_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('analyze_receipt 서비스 연결 실패.')
            return False, '', False

        req = AnalyzeReceipt.Request()
        req.image = self.camera_msg
        future = self.analyze_client.call_async(req)
        if not self._wait_future(future, timeout=30.0):
            return False, '', False
        resp = future.result()
        if resp is None:
            return False, '', False
        return resp.success, resp.destination, resp.press_bell

    # ----- 액션 전송 -----
    def send_approach_goal(self, destination, press_bell):
        """/approach_bell 액션 전송 후 결과까지 대기.
        반환: (success, message)"""
        if not self.approach_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('approach_bell 액션 서버 연결 실패.')
            return False, '서버 연결 실패'

        goal = ApproachBell.Goal()
        goal.destination = destination
        goal.press_bell = press_bell

        send_future = self.approach_client.send_goal_async(
            goal, feedback_callback=self._feedback_cb
        )
        if not self._wait_future(send_future):
            return False, '목표 전송 실패'
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, '목표 거부됨'

        result_future = goal_handle.get_result_async()
        # 시퀀스가 길 수 있으므로 넉넉히 대기
        if not self._wait_future(result_future, timeout=600.0):
            return False, '결과 대기 타임아웃'
        wrapped = result_future.result()
        if wrapped is None:
            return False, '결과 없음'
        return wrapped.result.success, wrapped.result.message

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        dist = fb.front_distance
        dist_str = f'{dist*100:.0f}cm' if dist >= 0 else 'N/A'
        self.get_logger().info(f'[액션 피드백] {fb.stage} (정면거리 {dist_str})')

    def _wait_future(self, future, timeout=60.0):
        start = time.time()
        while not future.done() and rclpy.ok():
            if time.time() - start > timeout:
                self.get_logger().error('future 대기 타임아웃.')
                return False
            time.sleep(0.05)
        return future.done()

    # =================================================================
    # 메인 루프 (메인 스레드에서 실행: OpenCV GUI 포함)
    # =================================================================
    def run_main_loop(self):
        print("\n========================================")
        print("  배달 로봇 시작")
        print("  영수증을 인식시켜 주세요. (호수 또는 경비실)")
        print("  Q / ESC: 전체 종료")
        print("========================================\n")

        mission_count = 0

        while rclpy.ok():
            # 1) 영수증 인식
            decision = capture_receipt_and_decide(
                self.get_frame, self.analyze_current_frame
            )
            if decision is None:
                print('\n=== 종료 요청. 루프 종료. ===')
                break

            destination, press_bell = decision
            mission_count += 1
            print(f"\n[미션 #{mission_count}] {destination}, 벨 누름={press_bell}")

            # 2) 목적지 이동 + 벨/회전 + 후진/정지
            success, message = self.send_approach_goal(destination, press_bell)
            print(f"  → 접근 결과: success={success}, {message}")

            # 3) F키 대기
            proceed = wait_for_f_key(
                window_title='Mission done - waiting',
                message='Mission done. Press F to go home.'
            )
            if not proceed:
                print('\n=== 종료 요청(F키 대기 중 Q/ESC). 루프 종료. ===')
                break

            # 4) 홈 복귀
            home_ok, home_msg = self.send_approach_goal('HOME', False)
            print(f"  → 홈 복귀: success={home_ok}, {home_msg}")

            print(f"\n=== 미션 #{mission_count} 완료. 다음 영수증 대기 ===\n")

        print(f'\n총 수행 미션: {mission_count}건')


def main():
    rclpy.init()
    node = StateMachineNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_main_loop()
    except KeyboardInterrupt:
        print('\n긴급 정지 (Ctrl+C)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
