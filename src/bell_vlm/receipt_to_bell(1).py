#!/usr/bin/env python3
"""영수증 → (몇 호 + 벨 여부) 판단 → 이동/벨 누르기 (단일 파일 실행)

사용 흐름:
1) 로봇 카메라(/camera/image_raw) 미리보기 창이 뜨면 영수증을 카메라에 보여주고
   SPACE(캡처)를 누른다.
   → 그 순간 프레임을 캡처해 VLM이 '몇 호'와 '벨을 눌러야 하는지'만 판단한다.
2) 판단 결과 창에서 물체를 올려둔 뒤 G(출발)를 누르면 동작을 수행한다.
   (R: 다시 캡처, Q: 취소)

동작 분기:
- 벨 누름  : 호수 좌표로 Nav2 이동 → VLM 정렬 → LiDAR 접근 → 짧은 전진으로 벨 누름
- 벨 안 누름: 호수 좌표로 Nav2 이동 후 정지

비고:
- 영수증 인식과 벨 정렬 모두 로봇 카메라(/camera/image_raw)를 사용한다.
- 발행 토픽은 /cmd_vel 하나뿐. 토픽/액션/서비스/모델명은 기존 그대로 유지.
- LiDAR 정면 기준은 scan 기준 180도(pi) 방향.

실행:
    python3 receipt_to_bell.py

환경변수:
    ANTHROPIC_API_KEY 필요 (터미널마다 export, 따옴표 없이)
"""

import math
import time
import json
import base64

import cv2
import numpy as np
import anthropic

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


# #####################################################################
# ##############  현장에서 확인 후 채워야 하는 부분  ###################
# #####################################################################
#
# 아직 목표 위치 좌표와 로봇 방향(yaw)을 모릅니다.
# 맵에서 각 호수(룸) 앞 위치를 확인한 뒤 아래 값을 채우세요.
#
#   x, y      : map 프레임 기준 목표 좌표 [m]
#   yaw_deg   : 목표 지점에서 로봇이 바라봐야 할 방향 [deg]
#               (벨을 눌러야 하는 방이면, 벨/벽을 정면으로 보도록 설정)
#
# 키(key)는 VLM이 영수증에서 읽어낼 '호수 숫자'와 일치시켜야 합니다.
# 예: 101호 → 101, 202호 → 202
#
ROOM_GOALS = {
    101: {'x': None, 'y': None, 'yaw_deg': None},   # TODO: 101호 좌표/방향 입력
    102: {'x': None, 'y': None, 'yaw_deg': None},   # TODO: 102호 좌표/방향 입력
    104: {'x': -0.3995890934060926, 'y': -0.4241495042894432, 'yaw_deg': 0.6433443218792397},   # TODO: 201호 좌표/방향 입력
    202: {'x': None, 'y': None, 'yaw_deg': None},   # TODO: 202호 좌표/방향 입력
    # 필요한 만큼 호수를 추가하세요.
}

# 목표 프레임 (보통 'map')
GOAL_FRAME_ID = 'map'
# #####################################################################
# #####################################################################


# =====================================================================
# 빨간색 검출 설정 (HSV)  — only_bell.py 그대로
# =====================================================================
RED_LOWER1 = np.array([0, 100, 80])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 100, 80])
RED_UPPER2 = np.array([179, 255, 255])

MIN_RED_AREA = 150
POSITION_CENTER_BAND = 0.10

ROTATE_GAIN = 0.5
ROTATE_MAX = 0.4
ROTATE_MIN = 0.08
ROTATE_MAX_FINE = 0.20
ROTATE_MIN_FINE = 0.05


# =====================================================================
# VLM 설정
# =====================================================================
VLM_MODEL = "claude-sonnet-4-6"
VLM_MAX_TOKENS = 256
JPEG_QUALITY = 85

# --- 영수증 판단용 프롬프트 ---
# 호수 + 벨 누름 여부만 JSON으로 받습니다.
RECEIPT_PROMPT = (
    "이 이미지는 배달 영수증입니다. 다음 두 가지만 판단해서 "
    "반드시 JSON 형식으로만 답하세요. 다른 설명은 절대 붙이지 마세요.\n"
    "1) room: 배달해야 할 호수 숫자 (예: 101, 202). 못 읽으면 null.\n"
    "2) press_bell: 벨(초인종)을 눌러야 하면 true, 아니면 false.\n"
    '형식 예시: {"room": 101, "press_bell": true}'
)

# --- 벨 정렬/검증용 프롬프트 (only_bell.py 그대로) ---
BELL_POSITION_PROMPT = (
    "이 이미지에 빨간색 벨 또는 둥근 빨간 버튼이 보이나요?\n"
    "반드시 다음 중 하나로만 답하세요:\n"
    "없음 / 왼쪽 / 중앙 / 오른쪽"
)
BELL_VERIFY_PROMPT = (
    "이 이미지 중앙 근처에 보이는 빨간색 물체가 "
    "눌러야 하는 빨간색 벨(또는 둥근 빨간 버튼)이 맞나요?\n"
    "'예' 또는 '아니오'로만 답하세요."
)
BELL_NEAR_PROMPT = (
    "빨간색 벨 또는 둥근 빨간 버튼이 가까이 보이나요? 간단히 답해주세요."
)


class BellVLM:
    """VLM 로직: 영수증 판단 + 벨 위치/검증/근접 확인"""

    def __init__(self, model=VLM_MODEL, logger=None):
        self.client = anthropic.Anthropic()
        self.model = model
        self._logger = logger

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(msg)
        else:
            print(msg)

    @staticmethod
    def ros_image_to_bgr(msg):
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )
        if msg.encoding == 'rgb8':
            return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        return raw

    @staticmethod
    def bgr_to_b64(bgr):
        success, jpeg = cv2.imencode(
            '.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not success:
            return None
        return base64.b64encode(jpeg.tobytes()).decode('utf-8')

    def ask(self, b64, question):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=VLM_MAX_TOKENS,
            messages=[
                {
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
                }
            ],
        )
        return response.content[0].text

    # ---------- 영수증 판단 ----------
    def read_receipt(self, bgr):
        """영수증 이미지 → {'room': int|None, 'press_bell': bool}

        실패 시 None 반환.
        """
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return None

        answer = self.ask(b64, RECEIPT_PROMPT)
        self._log(f'VLM 영수증 응답: {answer}')

        # JSON만 골라 파싱 (모델이 앞뒤로 텍스트를 붙였을 경우 대비)
        try:
            start = answer.index('{')
            end = answer.rindex('}') + 1
            data = json.loads(answer[start:end])
        except (ValueError, json.JSONDecodeError):
            self._log('영수증 JSON 파싱 실패')
            return None

        room = data.get('room')
        press = bool(data.get('press_bell', False))

        if room is not None:
            try:
                room = int(room)
            except (ValueError, TypeError):
                room = None

        return {'room': room, 'press_bell': press}

    # ---------- 벨 위치/검증/근접 (only_bell.py 그대로) ----------
    def bell_position(self, bgr):
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return '없음'
        answer = self.ask(b64, BELL_POSITION_PROMPT)
        if '중앙' in answer:
            return '중앙'
        elif '왼쪽' in answer:
            return '왼쪽'
        elif '오른쪽' in answer:
            return '오른쪽'
        else:
            return '없음'

    def is_bell(self, bgr):
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return False
        answer = self.ask(b64, BELL_VERIFY_PROMPT)
        self._log(f'VLM 벨 검증 응답: {answer}')
        return '예' in answer or 'yes' in answer.lower()

    def confirm_bell_near(self, bgr):
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return None
        return self.ask(b64, BELL_NEAR_PROMPT)


# =====================================================================
# 빨간색 검출기 (only_bell.py 그대로)
# =====================================================================
class RedBellDetector:
    def __init__(self, logger=None):
        self._logger = logger

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False, 0.0, 0, None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < MIN_RED_AREA:
            return False, 0.0, int(area), None

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return False, 0.0, int(area), None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        center_x = w / 2.0
        error = (cx - center_x) / (w / 2.0)

        return True, error, int(area), (cx, cy, w)

    @staticmethod
    def position_label(found, error):
        if not found:
            return '없음'
        if abs(error) <= POSITION_CENTER_BAND:
            return '있음-중앙'
        elif error < 0:
            return '있음-왼쪽'
        else:
            return '있음-오른쪽'


# =====================================================================
# 접근 설정 (only_bell.py 그대로)
# =====================================================================
APPROACH_STAGES = [
    {'target': 0.50, 'name': '50cm 정렬', 'tol': 0.10, 'confirm': 1, 'fine': False},
    {'target': 0.35, 'name': '35cm 접근', 'tol': 0.08, 'confirm': 1, 'fine': False},
    {'target': 0.25, 'name': '25cm 정밀정렬', 'tol': 0.03, 'confirm': 3, 'fine': True},
]

STOP_TOLERANCE = 0.02

FINAL_PUSH_DISTANCE = 0.08
FINAL_PUSH_SPEED = 0.02
FINAL_PUSH_TIME = FINAL_PUSH_DISTANCE / FINAL_PUSH_SPEED

# LiDAR 정면 기준: scan 기준 180도(pi) 방향
LIDAR_FRONT_ANGLE = 0
LIDAR_FRONT_WINDOW_DEG = 15.0

ALIGN_STEP_TIME = 0.15
ALIGN_STEP_TIME_FINE = 0.10

SEARCH_ANGULAR = 0.35
SEARCH_STEP_TIME = 0.4
SEARCH_FULL_TURN = 2.0 * math.pi

MAX_VERIFY_ATTEMPTS = 60


# =====================================================================
# 메인 제어 노드
# =====================================================================
class DeliveryController(Node):
    def __init__(self):
        super().__init__('delivery_controller')

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        self.camera_msg = None
        self.scan_msg = None

        self.cam_sub = self.create_subscription(
            Image, '/camera/image_raw', self._cam_cb, 1
        )

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, scan_qos
        )

        # Nav2 액션 클라이언트
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.detector = RedBellDetector(logger=self.get_logger())
        self.vlm = BellVLM(logger=self.get_logger())

    def _cam_cb(self, msg):
        self.camera_msg = msg

    def _scan_cb(self, msg):
        self.scan_msg = msg

    # ----- cmd_vel 헬퍼 -----
    def _pub_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
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

    # ----- 각도 헬퍼 -----
    def _normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _angle_diff(self, a, b):
        return abs(self._normalize_angle(a - b))

    @staticmethod
    def _yaw_to_quaternion(yaw_rad):
        """yaw(rad) → 쿼터니언 (z, w)만 사용"""
        return math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)

    # =================================================================
    # Nav2 이동
    # =================================================================
    def navigate_to_room(self, room):
        """ROOM_GOALS[room] 좌표로 Nav2 이동. 성공 True / 실패 False."""
        goal = ROOM_GOALS.get(room)
        if goal is None:
            self.get_logger().error(f'{room}호 좌표가 ROOM_GOALS에 없습니다.')
            return False
        if goal['x'] is None or goal['y'] is None or goal['yaw_deg'] is None:
            self.get_logger().error(
                f'{room}호 좌표/방향이 아직 입력되지 않았습니다(TODO). '
                f'ROOM_GOALS를 채우세요.'
            )
            return False

        self.get_logger().info('Nav2 서버 대기 중...')
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 액션 서버에 연결 실패.')
            return False

        yaw_rad = math.radians(goal['yaw_deg'])
        qz, qw = self._yaw_to_quaternion(yaw_rad)

        pose = PoseStamped()
        pose.header.frame_id = GOAL_FRAME_ID
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(goal['x'])
        pose.pose.position.y = float(goal['y'])
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = pose

        self.get_logger().info(
            f'{room}호로 이동 시작: x={goal["x"]}, y={goal["y"]}, '
            f'yaw={goal["yaw_deg"]}deg'
        )

        send_future = self.nav_client.send_goal_async(nav_goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Nav2가 목표를 거부했습니다.')
            return False

        self.get_logger().info('Nav2 목표 수락됨. 이동 중...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        if result is None:
            self.get_logger().error('Nav2 결과 수신 실패.')
            return False

        self.get_logger().info(f'{room}호 도착 완료.')
        return True

    # ----- LiDAR 정면 거리 -----
    def _get_front_distance(self):
        self.scan_msg = None
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.scan_msg is not None:
                break

        if self.scan_msg is None:
            self.get_logger().warn('scan_msg 수신 실패')
            return None

        scan = self.scan_msg
        if len(scan.ranges) == 0 or scan.angle_increment == 0.0:
            self.get_logger().warn('scan 데이터 이상')
            return None

        target_angle = LIDAR_FRONT_ANGLE
        angle_window = math.radians(LIDAR_FRONT_WINDOW_DEG)

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
            self.get_logger().warn(
                f'정면 ±{LIDAR_FRONT_WINDOW_DEG:.0f}도 범위 유효 거리 없음'
            )
            return None

        return min(valid)

    def _log_wall_distance(self, prefix=''):
        dist = self._get_front_distance()
        if dist is not None:
            self.get_logger().info(
                f'{prefix}벽(벨)과의 남은 거리: {dist*100:.1f}cm'
            )
        else:
            self.get_logger().warn(f'{prefix}거리 측정 실패')
        return dist

    # ----- 로봇 카메라 프레임 -----
    def _get_bgr(self):
        self.camera_msg = None
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.camera_msg is not None:
                break
        if self.camera_msg is None:
            return None
        return self.vlm.ros_image_to_bgr(self.camera_msg)

    # ----- VLM 위치 판별 로그 -----
    def _vlm_report_position(self, name):
        bgr = self._get_bgr()
        if bgr is None:
            self.get_logger().warn(f'[{name}] VLM 위치 확인용 이미지 없음')
            return '없음'
        pos = self.vlm.bell_position(bgr)
        if pos == '없음':
            self.get_logger().info(f'[{name}] VLM: 벨 미탐지')
        else:
            self.get_logger().info(f'[{name}] VLM: 벨 탐지 — 위치 {pos}')
        return pos

    # =================================================================
    # 빨간색 찾기 + VLM 벨 검증 (only_bell.py 그대로)
    # =================================================================
    def find_and_verify_bell(self, name):
        self.get_logger().info(f'----- [{name}] 빨간색 탐색 + VLM 벨 검증 -----')

        search_accum = 0.0

        for attempt in range(1, MAX_VERIFY_ATTEMPTS + 1):
            bgr = self._get_bgr()
            if bgr is None:
                self.get_logger().error(f'[{name}] 이미지 수신 실패. 재시도...')
                time.sleep(0.5)
                continue

            found, error, area, debug = self.detector.detect(bgr)
            label = self.detector.position_label(found, error)

            if found:
                self.get_logger().info(
                    f'[{name}] 빨간색 검출 ({label}, 오차={error:+.3f}, '
                    f'{area}px) → VLM 검증 요청'
                )

                if self.vlm.is_bell(bgr):
                    pos = self.vlm.bell_position(bgr)
                    self.get_logger().info(
                        f'[{name}] VLM: 벨 맞음 — 위치 {pos}. 정렬 진행.'
                    )
                    return True
                else:
                    self.get_logger().info(
                        f'[{name}] VLM: 이 빨간색은 벨 아님. 왼쪽으로 탐색.'
                    )
                    self._rotate(SEARCH_ANGULAR, SEARCH_STEP_TIME)
                    search_accum += SEARCH_ANGULAR * SEARCH_STEP_TIME
            else:
                self.get_logger().info(
                    f'[{name}] VLM: 벨 미탐지 (빨강 없음). 왼쪽으로 탐색 '
                    f'(누적 {math.degrees(search_accum):.0f}도)'
                )
                self._rotate(SEARCH_ANGULAR, SEARCH_STEP_TIME)
                search_accum += SEARCH_ANGULAR * SEARCH_STEP_TIME

            if search_accum >= SEARCH_FULL_TURN:
                self.get_logger().warn(
                    f'[{name}] 한 바퀴 돌았지만 벨 검증 실패. 다시 탐색.'
                )
                search_accum = 0.0

            time.sleep(0.2)

        self.get_logger().error(
            f'[{name}] 최대 {MAX_VERIFY_ATTEMPTS}회 탐색했지만 벨 검증 실패'
        )
        return False

    # =================================================================
    # 픽셀 기반 정밀 정렬 (only_bell.py 그대로)
    # =================================================================
    def align_to_bell(self, stage):
        name = stage['name']
        tol = stage['tol']
        need_confirm = stage['confirm']
        fine = stage['fine']

        rot_max = ROTATE_MAX_FINE if fine else ROTATE_MAX
        rot_min = ROTATE_MIN_FINE if fine else ROTATE_MIN
        step_time = ALIGN_STEP_TIME_FINE if fine else ALIGN_STEP_TIME

        self.get_logger().info(
            f'----- [{name}] 정렬 시작 | 허용오차 ±{tol*100:.0f}%, '
            f'확인 {need_confirm}회, {"정밀" if fine else "일반"} 모드 -----'
        )

        center_streak = 0

        while rclpy.ok():
            bgr = self._get_bgr()
            if bgr is None:
                self.get_logger().error('카메라 이미지 수신 실패. 재시도...')
                time.sleep(0.5)
                continue

            found, error, area, debug = self.detector.detect(bgr)
            label = self.detector.position_label(found, error)

            self._log_wall_distance(prefix=f'[{name}] ')

            if not found:
                self.get_logger().warn(
                    f'[{name}] 정렬 중 빨강 놓침. 재탐색 필요.'
                )
                self._stop()
                return False

            cx = debug[0] if debug else -1

            if abs(error) <= tol:
                center_streak += 1
                self.get_logger().info(
                    f'[{name}] 중앙 근처: {label}, 오차={error:+.3f} '
                    f'(연속 {center_streak}/{need_confirm})'
                )
                if center_streak >= need_confirm:
                    self._stop()
                    self.get_logger().info(
                        f'[{name}] 정렬 완료. 최종 오차 {error:+.3f}'
                    )
                    return True
                self._stop()
                time.sleep(0.15)
                continue

            center_streak = 0
            self.get_logger().info(
                f'[{name}] 빨강 위치: {label}, 오차={error:+.3f}, {area}px'
            )

            angular = -ROTATE_GAIN * error
            sign = 1.0 if angular >= 0 else -1.0
            mag = max(min(abs(angular), rot_max), rot_min)
            angular = sign * mag

            direction = '반시계(왼쪽)' if angular > 0 else '시계(오른쪽)'
            self.get_logger().info(
                f'[{name}] {direction} 회전 (angular_z={angular:+.3f})'
            )
            self._rotate(angular, step_time)
            time.sleep(0.1)

        self._stop()
        return False

    # =================================================================
    # 검증 + 정렬 묶음 (only_bell.py 그대로)
    # =================================================================
    def verify_and_align(self, stage):
        name = stage['name']

        while rclpy.ok():
            verified = self.find_and_verify_bell(name)
            if not verified:
                self.get_logger().error(f'[{name}] 벨 검증 실패. 단계 중단.')
                return False

            aligned = self.align_to_bell(stage)
            if aligned:
                return True

            self.get_logger().info(f'[{name}] 정렬 실패/놓침 → 검증부터 다시.')
            time.sleep(0.3)

        return False

    # =================================================================
    # LiDAR 기반 목표 거리까지 전진 (only_bell.py 그대로)
    # =================================================================
    def approach_to_distance(self, target_distance, name=''):
        self.get_logger().info(
            f'----- [{name}] {target_distance*100:.0f}cm까지 이동 -----'
        )

        while rclpy.ok():
            dist = self._get_front_distance()
            if dist is None:
                self.get_logger().warn('LiDAR 거리 측정 실패. 재시도...')
                self._stop()
                time.sleep(0.5)
                continue

            remaining = dist - target_distance
            self.get_logger().info(
                f'[{name}] 벽(벨)과의 남은 거리: {dist*100:.1f}cm '
                f'(목표 {target_distance*100:.0f}cm, 더 갈 거리 {remaining*100:.1f}cm)'
            )

            if remaining <= STOP_TOLERANCE:
                self._stop()
                self.get_logger().info(
                    f'[{name}] 목표 도달. 현재 {dist*100:.1f}cm'
                )
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
    # 버튼 누르기 전진 (only_bell.py 그대로)
    # =================================================================
    def final_push(self):
        dist_before = self._log_wall_distance(prefix='[버튼누르기 전] ')

        self.get_logger().info(
            f'----- 버튼 누르기 전진: {FINAL_PUSH_DISTANCE*100:.0f}cm '
            f'({FINAL_PUSH_SPEED}m/s × {FINAL_PUSH_TIME:.1f}s) -----'
        )

        push_end_time = time.time() + FINAL_PUSH_TIME
        next_log = time.time()

        while time.time() < push_end_time and rclpy.ok():
            self._pub_cmd(FINAL_PUSH_SPEED, 0.0)
            if time.time() >= next_log:
                self.get_logger().info(
                    f'전진 중... 남은 시간 {push_end_time - time.time():.1f}s'
                )
                next_log = time.time() + 0.5
            time.sleep(0.05)

        self._stop()

        dist_after = self._log_wall_distance(prefix='[버튼누르기 후] ')
        if dist_before is not None and dist_after is not None:
            self.get_logger().info(
                f'실제 전진 거리(추정): {(dist_before - dist_after)*100:.1f}cm'
            )
        self.get_logger().info('버튼 누르기 전진 완료. 정지.')

    # =================================================================
    # 벨 누르기 전체 시퀀스 (only_bell.py run()의 벨 부분)
    # =================================================================
    def press_bell_sequence(self):
        self.get_logger().info('===== 벨 누르기 시퀀스 시작 =====')

        for idx, stage in enumerate(APPROACH_STAGES):
            name = stage['name']
            self.get_logger().info(
                f'### 단계 {idx+1}/{len(APPROACH_STAGES)}: {name} ###'
            )

            self._vlm_report_position(name)

            if idx > 0:
                if not self.approach_to_distance(stage['target'], name):
                    self.get_logger().error(f'[{name}] 이동 중단')
                    return False

            if not self.verify_and_align(stage):
                self.get_logger().error(f'[{name}] 검증/정렬 실패')
                return False

        self.final_push()

        bgr = self._get_bgr()
        if bgr is not None:
            answer = self.vlm.confirm_bell_near(bgr)
            final_dist = self._get_front_distance()
            print("\n=== 최종 상태 ===")
            if final_dist is not None:
                print(f"벽(벨)과의 거리: {final_dist*100:.1f}cm")
            print(f"VLM 판단: {answer}")

        return True

    # =================================================================
    # 영수증 판단 결과를 받아 미션 수행
    # =================================================================
    def execute_mission(self, room, press_bell):
        self.get_logger().info(
            f'===== 미션 시작: {room}호, 벨 누름={press_bell} ====='
        )

        # 1) 호수로 이동
        if not self.navigate_to_room(room):
            self.get_logger().error('이동 실패. 미션 중단.')
            return False

        # 2) 분기
        if press_bell:
            return self.press_bell_sequence()
        else:
            self.get_logger().info('벨 누름 불필요. 도착 후 정지.')
            self._stop()
            return True


# =====================================================================
# 로봇 카메라(/camera/image_raw)로 영수증 캡처 + 키 입력 UI
# =====================================================================
def capture_receipt_and_decide(node):
    """로봇 카메라 미리보기 → SPACE 캡처 → VLM 판단
       → G 출발 / R 재캡처 / Q 취소

    반환: (room, press_bell) 또는 None(취소)
    """
    print("\n[로봇 카메라] 영수증을 카메라에 보여주고 SPACE로 캡처하세요. (Q: 종료)")

    decision = None
    state = 'preview'   # preview → decided
    last_capture = None  # decided 상태에서 멈춰 보여줄 캡처 프레임

    try:
        while rclpy.ok():
            if state == 'preview':
                # 로봇 카메라에서 최신 프레임 받기
                frame = node._get_bgr()
                if frame is None:
                    print('로봇 카메라 프레임 수신 실패. '
                          '토픽(/camera/image_raw) 확인 중...')
                    time.sleep(0.3)
                    continue
            else:
                # decided 상태: 캡처한 프레임을 그대로 유지
                frame = last_capture

            view = frame.copy()
            if state == 'preview':
                cv2.putText(view, 'SPACE: capture  Q: quit',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            else:
                room, press = decision['room'], decision['press_bell']
                cv2.putText(view, f'room={room}  press_bell={press}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                cv2.putText(view, 'G: go  R: recapture  Q: quit',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            cv2.imshow('Receipt', view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                decision = None
                break

            if state == 'preview' and key == ord(' '):
                print('[캡처] 영수증 분석 중...')
                last_capture = frame.copy()
                result = node.vlm.read_receipt(last_capture)
                if result is None or result['room'] is None:
                    print('판단 실패(호수 인식 불가). 다시 시도하세요.')
                    continue
                decision = result
                print(f"=> 판단 결과: {decision['room']}호, "
                      f"벨 누름={decision['press_bell']}")
                state = 'decided'

            elif state == 'decided' and key == ord('r'):
                print('다시 캡처 모드로 전환.')
                decision = None
                last_capture = None
                state = 'preview'

            elif state == 'decided' and key == ord('g'):
                print('[출발]')
                break

    finally:
        cv2.destroyAllWindows()

    if decision is None:
        return None
    return decision['room'], decision['press_bell']


def main():
    rclpy.init()
    node = DeliveryController()

    try:
        decision = capture_receipt_and_decide(node)
        if decision is None:
            print('\n=== 취소됨 ===')
            return

        room, press_bell = decision
        if node.execute_mission(room, press_bell):
            print("\n=== 전체 미션 완료 ===")
        else:
            print("\n=== 미션 중단 ===")

    except KeyboardInterrupt:
        node._stop()
        print('\n긴급 정지')
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
