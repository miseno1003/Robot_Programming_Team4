#!/usr/bin/env python3
"""nav_node: 네비게이션 + 벨 접근/누르기 + 모터 제어 노드.

제공 인터페이스:
- 액션 /approach_bell (delivery_interfaces/ApproachBell)
    목적지로 Nav2 이동 후, 벨 누름/미누름 시퀀스 수행.
    destination="HOME" 이면 초기 위치로만 복귀.
- 토픽 /delivery_status (std_msgs/String)  [발행]
    현재 진행 상황을 사람이 읽을 수 있는 문자열로 알림.

사용 인터페이스(클라이언트):
- 서비스 /verify_bell  → vlm_node 에 벨 검증/위치 요청
- 액션 navigate_to_pose → Nav2 로 이동

토픽 구독:
- /camera/image_raw (sensor_msgs/Image)
- /scan (sensor_msgs/LaserScan)
토픽 발행:
- /cmd_vel (geometry_msgs/TwistStamped)
"""

import math
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose

from delivery_interfaces.action import ApproachBell
from delivery_interfaces.srv import VerifyBell

from delivery_nav.red_detector import RedBellDetector
from delivery_nav import config as cfg


def imgmsg_to_bgr(msg):
    """sensor_msgs/Image → OpenCV BGR ndarray (cv_bridge 없이 수동 변환)."""
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    return raw.copy()


class NavNode(Node):
    def __init__(self):
        super().__init__("nav_node")

        self.detector = RedBellDetector(logger=self.get_logger())

        self.camera_msg = None
        self.scan_msg = None

        self._cb = ReentrantCallbackGroup()

        # ---- 발행 ----
        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/delivery_status", 10)

        # ---- 구독 ----
        self.cam_sub = self.create_subscription(
            Image, "/camera/image_raw", self._cam_cb, 1, callback_group=self._cb
        )
        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self._scan_cb, scan_qos, callback_group=self._cb
        )

        # ---- 클라이언트 ----
        self.nav_client = ActionClient(
            self, NavigateToPose, "navigate_to_pose", callback_group=self._cb
        )
        self.verify_client = self.create_client(
            VerifyBell, "verify_bell", callback_group=self._cb
        )

        # ---- 액션 서버 ----
        self.action_server = ActionServer(
            self,
            ApproachBell,
            "approach_bell",
            execute_callback=self.execute_callback,
            callback_group=self._cb,
        )

        self._goal_handle = None  # 피드백 발행용

        self.get_logger().info("nav_node 준비 완료. (액션 /approach_bell)")

    # =================================================================
    # 콜백 / 상태 발행
    # =================================================================
    def _cam_cb(self, msg):
        self.camera_msg = msg

    def _scan_cb(self, msg):
        self.scan_msg = msg

    def _publish_status(self, text):
        self.status_pub.publish(String(data=text))
        self.get_logger().info(f"[status] {text}")

    def _publish_feedback(self, stage, dist=-1.0):
        if self._goal_handle is not None:
            fb = ApproachBell.Feedback()
            fb.stage = stage
            fb.front_distance = float(dist if dist is not None else -1.0)
            self._goal_handle.publish_feedback(fb)
        self._publish_status(stage)

    # =================================================================
    # 액션 실행 콜백
    # =================================================================
    def execute_callback(self, goal_handle):
        self._goal_handle = goal_handle
        destination = goal_handle.request.destination
        press_bell = goal_handle.request.press_bell

        result = ApproachBell.Result()

        # ---- HOME 복귀만 ----
        if destination == "HOME":
            self._publish_feedback("홈으로 복귀 시작")
            ok = self.navigate_to_home()
            goal_handle.succeed()
            result.success = ok
            result.message = "홈 복귀 완료" if ok else "홈 복귀 실패"
            self._goal_handle = None
            return result

        # ---- 목적지 이동 ----
        self._publish_feedback(f"{destination} 이동 시작")
        if not self.navigate_to_destination(destination):
            self._publish_feedback(f"{destination} 이동 실패")
            goal_handle.succeed()
            result.success = False
            result.message = f"{destination} 이동 실패"
            self._goal_handle = None
            return result

        # ---- 벨 누름 / 미누름 분기 ----
        if press_bell:
            self._publish_feedback("벨 누르기 시퀀스 시작")
            seq_ok = self.press_bell_sequence()
            self.return_after_bell()
            result.success = seq_ok
            result.message = "벨 누름 + 후진 완료" if seq_ok else "벨 시퀀스 실패"
        else:
            self._publish_feedback("벨 미누름: 도착 후 180도 회전")
            self.return_after_no_bell()
            result.success = True
            result.message = "도착 + 회전 완료"

        goal_handle.succeed()
        self._goal_handle = None
        return result

    # =================================================================
    # cmd_vel 헬퍼
    # =================================================================
    def _pub_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def _stop(self):
        self._pub_cmd(0.0, 0.0)

    def _rotate(self, angular_z, duration):
        end_time = time.time() + duration
        while time.time() < end_time and rclpy.ok():
            self._pub_cmd(0.0, angular_z)
            time.sleep(0.05)
        self._stop()
        time.sleep(0.2)

    # =================================================================
    # 각도 헬퍼
    # =================================================================
    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _angle_diff(self, a, b):
        return abs(self._normalize_angle(a - b))

    @staticmethod
    def _yaw_to_quaternion(yaw_rad):
        return math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)

    # =================================================================
    # Nav2 이동
    # =================================================================
    def _navigate_to_pose(self, pose_dict, label="목표"):
        if pose_dict["x"] is None or pose_dict["y"] is None:
            self.get_logger().error(f"{label} 좌표가 None 입니다.")
            return False

        self.get_logger().info("Nav2 서버 대기 중...")
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Nav2 액션 서버 연결 실패.")
            return False

        yaw_rad = math.radians(pose_dict.get("yaw_deg", 0.0))
        qz, qw = self._yaw_to_quaternion(yaw_rad)

        pose = PoseStamped()
        pose.header.frame_id = cfg.GOAL_FRAME_ID
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(pose_dict["x"])
        pose.pose.position.y = float(pose_dict["y"])
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        nav_goal = NavigateToPose.Goal()  # Nav2 action 목표 생성
        nav_goal.pose = pose

        self.get_logger().info(
            f'[{label}] 이동 시작: x={pose_dict["x"]}, y={pose_dict["y"]}, '
            f'yaw={pose_dict.get("yaw_deg", 0.0)}deg'
        )

        send_future = self.nav_client.send_goal_async(nav_goal)  # Nav2 목표 전송
        if not self._wait_future(send_future):
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Nav2 가 목표를 거부했습니다.")
            return False

        self.get_logger().info(f"[{label}] Nav2 목표 수락됨. 이동 중...")
        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future):
            return False
        if result_future.result() is None:
            self.get_logger().error("Nav2 결과 수신 실패.")
            return False

        self.get_logger().info(f"[{label}] 도착 완료.")
        return True

    def _wait_future(self, future, timeout=120.0):
        """MultiThreadedExecutor 가 다른 스레드에서 future 를 완료시킨다.
        여기서는 done() 만 폴링한다 (executor 재진입 금지)."""
        start = time.time()
        while not future.done() and rclpy.ok():
            if time.time() - start > timeout:
                self.get_logger().error("future 대기 타임아웃.")
                return False
            time.sleep(0.05)
        return future.done()

    def navigate_to_destination(self, destination):
        if destination == "경비실":
            if cfg.GUARD_ROOM_GOAL["x"] is None:
                self.get_logger().error("경비실 좌표 미입력(TODO).")
                return False
            return self._navigate_to_pose(cfg.GUARD_ROOM_GOAL, label="경비실")

        try:
            room = int(destination)
        except (ValueError, TypeError):
            self.get_logger().error(f"알 수 없는 목적지: {destination}")
            return False

        goal = cfg.ROOM_GOALS.get(room)
        if goal is None:
            self.get_logger().error(f"{room}호 좌표가 ROOM_GOALS 에 없습니다.")
            return False
        if goal["x"] is None or goal["y"] is None or goal["yaw_deg"] is None:
            self.get_logger().error(f"{room}호 좌표/방향 미입력(TODO).")
            return False
        return self._navigate_to_pose(goal, label=f"{room}호")

    def navigate_to_home(self):
        self.get_logger().info("===== 홈(초기 위치) 복귀 시작 =====")
        ok = self._navigate_to_pose(cfg.HOME_POSITION, label="홈")
        self._publish_status("홈 복귀 완료" if ok else "홈 복귀 실패")
        return ok

    # =================================================================
    # LiDAR 정면 거리
    # =================================================================
    def _get_front_distance(self):
        if self.scan_msg is None:
            return None
        scan = self.scan_msg
        if len(scan.ranges) == 0 or scan.angle_increment == 0.0:
            return None

        target_angle = cfg.LIDAR_FRONT_ANGLE
        angle_window = math.radians(cfg.LIDAR_FRONT_WINDOW_DEG)

        valid = []
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r):
                continue
            if not (scan.range_min < r < scan.range_max):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if self._angle_diff(angle, target_angle) <= angle_window:
                valid.append(r)
        if not valid:
            return None
        return min(valid)

    def _log_wall_distance(self, prefix=""):
        dist = self._get_front_distance()
        if dist is not None:
            self.get_logger().info(f"{prefix}벽(벨)과의 거리: {dist*100:.1f}cm")
        return dist

    # =================================================================
    # 카메라 프레임
    # =================================================================
    def _get_bgr(self, timeout=5.0):
        start = time.time()
        while self.camera_msg is None and rclpy.ok():
            if time.time() - start > timeout:
                return None
            time.sleep(0.05)
        try:
            return imgmsg_to_bgr(self.camera_msg)
        except Exception as e:
            self.get_logger().error(f"카메라 변환 실패: {e}")
            return None

    # =================================================================
    # /verify_bell 서비스 호출
    # =================================================================
    def _verify_bell_via_service(self):
        """현재 카메라 이미지를 vlm_node 로 보내 벨 검증.
        반환: (is_bell, position)"""
        if self.camera_msg is None:
            return False, "없음"
        if not self.verify_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("verify_bell 서비스 연결 실패.")
            return False, "없음"

        req = VerifyBell.Request()
        req.image = self.camera_msg
        future = self.verify_client.call_async(req)
        if not self._wait_future(future, timeout=30.0):
            return False, "없음"
        resp = future.result()
        if resp is None:
            return False, "없음"
        return resp.is_bell, resp.position

    # =================================================================
    # 빨간색 찾기 + VLM 벨 검증
    # =================================================================
    def find_and_verify_bell(self, name):
        self.get_logger().info(f"----- [{name}] 빨간색 탐색 + VLM 검증 -----")
        search_accum = 0.0

        for attempt in range(1, cfg.MAX_VERIFY_ATTEMPTS + 1):
            bgr = self._get_bgr()
            if bgr is None:
                self.get_logger().error(f"[{name}] 이미지 수신 실패. 재시도...")
                time.sleep(0.5)
                continue

            found, error, area, debug = self.detector.detect(bgr)
            label = self.detector.position_label(found, error)

            if found:
                self.get_logger().info(
                    f"[{name}] 빨간색 검출 ({label}, 오차={error:+.3f}, "
                    f"{area}px) → VLM 검증 요청"
                )
                is_bell, pos = self._verify_bell_via_service()
                if is_bell:
                    self.get_logger().info(
                        f"[{name}] VLM: 벨 맞음(위치 {pos}). 정렬 진행."
                    )
                    return True
                else:
                    self.get_logger().info(f"[{name}] VLM: 벨 아님. 왼쪽 탐색.")
                    self._rotate(cfg.SEARCH_ANGULAR, cfg.SEARCH_STEP_TIME)
                    search_accum += cfg.SEARCH_ANGULAR * cfg.SEARCH_STEP_TIME
            else:
                self.get_logger().info(
                    f"[{name}] 빨강 없음. 왼쪽 탐색 "
                    f"(누적 {math.degrees(search_accum):.0f}도)"
                )
                self._rotate(cfg.SEARCH_ANGULAR, cfg.SEARCH_STEP_TIME)
                search_accum += cfg.SEARCH_ANGULAR * cfg.SEARCH_STEP_TIME

            if search_accum >= cfg.SEARCH_FULL_TURN:
                self.get_logger().warn(f"[{name}] 한 바퀴 돌았지만 실패. 다시 탐색.")
                search_accum = 0.0
            time.sleep(0.2)

        self.get_logger().error(f"[{name}] 최대 탐색 횟수 초과. 벨 검증 실패.")
        return False

    # =================================================================
    # 픽셀 기반 정밀 정렬
    # =================================================================
    def align_to_bell(self, stage):
        name = stage["name"]
        tol = stage["tol"]
        need_confirm = stage["confirm"]
        fine = stage["fine"]

        rot_max = cfg.ROTATE_MAX_FINE if fine else cfg.ROTATE_MAX
        rot_min = cfg.ROTATE_MIN_FINE if fine else cfg.ROTATE_MIN
        step_time = cfg.ALIGN_STEP_TIME_FINE if fine else cfg.ALIGN_STEP_TIME

        self.get_logger().info(
            f"----- [{name}] 정렬 시작 | 허용오차 ±{tol*100:.0f}%, "
            f'확인 {need_confirm}회, {"정밀" if fine else "일반"} -----'
        )
        center_streak = 0

        while rclpy.ok():
            bgr = self._get_bgr()
            if bgr is None:
                self.get_logger().error("이미지 수신 실패. 재시도...")
                time.sleep(0.5)
                continue

            found, error, area, debug = self.detector.detect(bgr)
            label = self.detector.position_label(found, error)
            self._log_wall_distance(prefix=f"[{name}] ")

            if not found:
                self.get_logger().warn(f"[{name}] 정렬 중 빨강 놓침. 재탐색 필요.")
                self._stop()
                return False

            if abs(error) <= tol:
                center_streak += 1
                self.get_logger().info(
                    f"[{name}] 중앙 근처: {label}, 오차={error:+.3f} "
                    f"(연속 {center_streak}/{need_confirm})"
                )
                if center_streak >= need_confirm:
                    self._stop()
                    self.get_logger().info(f"[{name}] 정렬 완료. 오차 {error:+.3f}")
                    return True
                self._stop()
                time.sleep(0.15)
                continue

            center_streak = 0
            angular = -cfg.ROTATE_GAIN * error
            sign = 1.0 if angular >= 0 else -1.0
            mag = max(min(abs(angular), rot_max), rot_min)
            angular = sign * mag
            direction = "반시계(왼쪽)" if angular > 0 else "시계(오른쪽)"
            self.get_logger().info(
                f"[{name}] {direction} 회전 (angular_z={angular:+.3f}), 오차={error:+.3f}"
            )
            self._rotate(angular, step_time)
            time.sleep(0.1)

        self._stop()
        return False

    # =================================================================
    # 검증 + 정렬 묶음
    # =================================================================
    def verify_and_align(self, stage):
        name = stage["name"]
        while rclpy.ok():
            if not self.find_and_verify_bell(name):
                self.get_logger().error(f"[{name}] 벨 검증 실패. 단계 중단.")
                return False
            if self.align_to_bell(stage):
                return True
            self.get_logger().info(f"[{name}] 정렬 실패/놓침 → 검증부터 다시.")
            time.sleep(0.3)
        return False

    # =================================================================
    # LiDAR 기반 목표 거리까지 전진
    # =================================================================
    def approach_to_distance(self, target_distance, name=""):
        self.get_logger().info(
            f"----- [{name}] {target_distance*100:.0f}cm까지 이동 -----"
        )
        while rclpy.ok():
            dist = self._get_front_distance()
            if dist is None:
                self.get_logger().warn("LiDAR 거리 측정 실패. 재시도...")
                self._stop()
                time.sleep(0.5)
                continue

            remaining = dist - target_distance
            self.get_logger().info(
                f"[{name}] 남은 거리: {dist*100:.1f}cm "
                f"(목표 {target_distance*100:.0f}cm, 더 갈 거리 {remaining*100:.1f}cm)"
            )
            if remaining <= cfg.STOP_TOLERANCE:
                self._stop()
                self.get_logger().info(f"[{name}] 목표 도달. 현재 {dist*100:.1f}cm")
                return True

            if remaining > 0.20:
                speed = 0.06
            elif remaining > 0.10:
                speed = 0.03
            else:
                speed = 0.02
            self._pub_cmd(speed, 0.0)
            time.sleep(0.1)

        self._stop()
        return False

    # =================================================================
    # 버튼 누르기 전진
    # =================================================================
    def final_push(self):
        dist_before = self._log_wall_distance(prefix="[버튼누르기 전] ")
        self.get_logger().info(
            f"----- 버튼 누르기 전진: {cfg.FINAL_PUSH_DISTANCE*100:.0f}cm "
            f"({cfg.FINAL_PUSH_SPEED}m/s × {cfg.FINAL_PUSH_TIME:.1f}s) -----"
        )
        push_end_time = time.time() + cfg.FINAL_PUSH_TIME
        while time.time() < push_end_time and rclpy.ok():
            self._pub_cmd(cfg.FINAL_PUSH_SPEED, 0.0)
            time.sleep(0.05)
        self._stop()

        dist_after = self._log_wall_distance(prefix="[버튼누르기 후] ")
        if dist_before is not None and dist_after is not None:
            self.get_logger().info(
                f"실제 전진 거리(추정): {(dist_before - dist_after)*100:.1f}cm"
            )
        self.get_logger().info("버튼 누르기 전진 완료. 정지.")

    # =================================================================
    # 벨 누르기 전체 시퀀스
    # =================================================================
    def press_bell_sequence(self):
        self.get_logger().info("===== 벨 누르기 시퀀스 시작 =====")
        for idx, stage in enumerate(cfg.APPROACH_STAGES):
            name = stage["name"]
            self._publish_feedback(
                f"단계 {idx+1}/{len(cfg.APPROACH_STAGES)}: {name}",
                self._get_front_distance(),
            )
            if idx > 0:
                if not self.approach_to_distance(stage["target"], name):
                    self.get_logger().error(f"[{name}] 이동 중단")
                    return False
            if not self.verify_and_align(stage):
                self.get_logger().error(f"[{name}] 검증/정렬 실패")
                return False

        self.final_push()
        self._publish_feedback("벨 누르기 완료", self._get_front_distance())
        return True

    # =================================================================
    # 벨 누른 후 복귀 (정지 + 후진). 홈 이동은 별도 HOME 액션으로 처리.
    # =================================================================
    def return_after_bell(self):
        self.get_logger().info(
            f"===== 벨 눌림 후 | {cfg.BELL_PRESSED_STOP_SEC:.0f}초 정지 ====="
        )
        self._stop()
        time.sleep(cfg.BELL_PRESSED_STOP_SEC)

        reverse_time = cfg.REVERSE_DISTANCE_M / cfg.REVERSE_SPEED
        self.get_logger().info(
            f"후진: {cfg.REVERSE_DISTANCE_M*100:.0f}cm "
            f"({cfg.REVERSE_SPEED}m/s × {reverse_time:.1f}s)"
        )
        end_t = time.time() + reverse_time
        while time.time() < end_t and rclpy.ok():
            self._pub_cmd(-cfg.REVERSE_SPEED, 0.0)
            time.sleep(0.05)
        self._stop()
        self._publish_feedback("벨 + 후진 완료. F키 대기 준비")

    # =================================================================
    # 벨 미누름 후 복귀 (180도 회전 + 정지)
    # =================================================================
    def return_after_no_bell(self):
        half_turn_time = math.pi / cfg.HALF_ROTATE_ANGULAR
        self.get_logger().info(
            f"===== 제자리 180도 회전: "
            f"{cfg.HALF_ROTATE_ANGULAR}rad/s × {half_turn_time:.1f}s ====="
        )
        end_t = time.time() + half_turn_time
        while time.time() < end_t and rclpy.ok():
            self._pub_cmd(0.0, cfg.HALF_ROTATE_ANGULAR)
            time.sleep(0.05)
        self._stop()
        self.get_logger().info(f"180도 회전 완료. {cfg.NO_BELL_STOP_SEC:.0f}초 정지.")
        time.sleep(cfg.NO_BELL_STOP_SEC)
        self._publish_feedback("도착 + 회전 완료. F키 대기 준비")


def main():
    rclpy.init()
    node = NavNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
