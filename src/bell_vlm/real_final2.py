#!/usr/bin/env python3
"""영수증 → (몇 호 + 벨 여부) 판단 → 이동/벨 누르기 → 초기 위치 복귀 → 반복

사용 흐름:
1) 로봇 카메라(/camera/image_raw) 미리보기 창이 뜨면 영수증을 카메라에 보여주고
   SPACE(캡처)를 누른다.
   → 그 순간 프레임을 캡처해 VLM이 '몇 호'와 '벨을 눌러야 하는지'만 판단한다.
   → '경비실'로 인식되면 경비실 좌표로 이동한다.
2) 판단 결과 창에서 물체를 올려둔 뒤 G(출발)를 누르면 동작을 수행한다.
   (R: 다시 캡처, Q: 취소)

동작 분기:
- 벨 누름  : 호수/경비실 좌표로 Nav2 이동 → VLM 정렬 → LiDAR 접근 → 짧은 전진으로 벨 누름
             → 3초 정지 → 15cm 후진 → [F키 대기] → 초기 위치로 Nav2 복귀
- 벨 안 누름: 호수/경비실 좌표로 Nav2 이동 후 정지
             → 제자리 180도 회전 → 2초 정지 → [F키 대기] → 초기 위치로 Nav2 복귀

복귀 후 다시 영수증 인식 대기 (루프). ESC 또는 Q 키를 누르면 전체 종료.

비고:
- 영수증 인식과 벨 정렬 모두 로봇 카메라(/camera/image_raw)를 사용한다.
- 발행 토픽은 /cmd_vel 하나뿐. (위치 검증용으로 /amcl_pose는 구독만 한다.)
- LiDAR 정면 기준은 scan 기준 0도 방향.
- 경비실은 room 값이 '경비실' 문자열로 인식된 경우 GUARD_ROOM_GOAL 좌표로 이동.

[도착 위치 검증]
- Nav2가 "도착(success)"을 반환해도 실제 위치는 목표와 어긋날 수 있다.
- 그래서 Nav2 완료 직후 /amcl_pose로 현재 위치를 읽어 목표와의 거리/각도 오차를
  직접 계산하고, 허용 범위를 벗어나면 같은 목표로 재주행한다(최대 MAX_NAV_RETRIES회).
- AMCL을 받지 못하면(미실행 등) 기존처럼 도착으로 간주하는 안전 폴백을 둔다.

실행:
    python3 receipt_to_bell_2_.py

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
from geometry_msgs.msg import TwistStamped, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


# #####################################################################
# ##############         초기(홈) 위치 설정          ##################
# #####################################################################
#
# 로봇이 모든 미션 완료 후 반드시 돌아올 위치입니다.
# 맵에서 실제 시작 위치를 확인한 뒤 아래 값을 채우세요.
#
HOME_POSITION = {
    'x': -0.16646553628836436,
    'y': -0.03660610737847862,
    'yaw_deg': 0,
}

# #####################################################################
# ##############         경비실 위치 설정            ##################
# #####################################################################
#
# VLM이 영수증에서 '경비실'을 인식하면 이 좌표로 이동합니다.
# 맵에서 경비실 앞 위치를 확인한 뒤 아래 값을 채우세요.
#
#   x, y      : map 프레임 기준 목표 좌표 [m]
#   yaw_deg   : 목표 지점에서 로봇이 바라봐야 할 방향 [deg]
#
GUARD_ROOM_GOAL = {
    'x': -0.8286276588680647,
    'y': -0.03660610737847862,
    'yaw_deg': 180, # TODO: 경비실 방향(deg) 입력
}

# #####################################################################
# ##############  현장에서 확인 후 채워야 하는 부분  ###################
# #####################################################################
#
# 맵에서 각 호수(룸) 앞 위치를 확인한 뒤 아래 값을 채우세요.
#
# 키(key)는 VLM이 영수증에서 읽어낼 '호수 숫자'와 일치시켜야 합니다.
# 예: 101호 → 101, 202호 → 202
#
ROOM_GOALS = {
    102: {'x': -1.1335877556762092, 'y': 0.79288580447868, 'yaw_deg': 80},
    103: {'x': -1.03544384739192, 'y': -0.08952725350345325, 'yaw_deg': 180},
    105: {'x': --1.8558374400489206, 'y': -0.5882756015382987, 'yaw_deg': -90},
    # 필요한 만큼 호수를 추가하세요.
}

# 목표 프레임 (보통 'map')
GOAL_FRAME_ID = 'map'
# #####################################################################
# #####################################################################


# =====================================================================
# 도착 위치 검증(피드백) 설정
# =====================================================================
# Nav2가 "도착"을 반환해도 /amcl_pose로 실제 위치를 재확인한다.
GOAL_DISTANCE_TOLERANCE = 0.2   # 목표와의 허용 거리 오차 [m]
GOAL_YAW_TOLERANCE_DEG = 25.0    # 목표와의 허용 각도 오차 [deg]
MAX_NAV_RETRIES = 15             # 위치 오차 초과 시 추가 재주행 횟수
AMCL_WAIT_SPINS = 15            # amcl_pose 최신화를 위한 spin_once 반복 횟수


# =====================================================================
# 복귀 관련 설정
# =====================================================================
# 벨을 누른 경우: 정지 시간 [초]
BELL_PRESSED_STOP_SEC = 3.0

# 벨을 누른 경우: 후진 거리/속도
REVERSE_DISTANCE_M = 0.15
REVERSE_SPEED = 0.05         # [m/s] (양수, 내부에서 음수로 적용)

# 벨 안 누른 경우: 180도 회전 후 정지 시간 [초]
NO_BELL_STOP_SEC = 2.0

# 벨 안 누른 경우: 제자리 180도 회전 속도 (벨 후진과 동일 속도 사용)
HALF_ROTATE_ANGULAR = 0.5   # [rad/s]


# =====================================================================
# 빨간색 검출 설정 (HSV)
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
# room 값: 호수 숫자(예: 101) 또는 "경비실" 문자열
RECEIPT_PROMPT = (
    "이 이미지는 배달 영수증입니다. 다음 두 가지만 판단해서 "
    "반드시 JSON 형식으로만 답하세요. 다른 설명은 절대 붙이지 마세요.\n"
    "1) room: 배달해야 할 호수 숫자 (예: 101, 202) 또는 경비실이면 \"경비실\" 문자열. 못 읽으면 null.\n"
    "2) press_bell: 벨(초인종)을 눌러야 하면 true, 아니면 false.\n"
    '형식 예시(호수): {"room": 101, "press_bell": true}\n'
    '형식 예시(경비실): {"room": "경비실", "press_bell": false}'
)

# --- 벨 정렬/검증용 프롬프트 ---
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

# 경비실로 판단하는 키워드 (VLM 응답에서 이 중 하나라도 포함되면 경비실로 처리)
GUARD_ROOM_KEYWORDS = ['경비실', '경비', 'guard', 'security']


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
        """영수증 이미지 → {'room': int|str|None, 'press_bell': bool}

        room이 경비실 키워드면 '경비실' 문자열로 정규화.
        실패 시 None 반환.
        """
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return None

        answer = self.ask(b64, RECEIPT_PROMPT)
        self._log(f'VLM 영수증 응답: {answer}')

        try:
            start = answer.index('{')
            end = answer.rindex('}') + 1
            data = json.loads(answer[start:end])
        except (ValueError, json.JSONDecodeError):
            self._log('영수증 JSON 파싱 실패')
            return None

        room = data.get('room')
        press = bool(data.get('press_bell', False))

        # 경비실 키워드 판별
        if room is not None:
            room_str = str(room).strip()
            if any(kw in room_str for kw in GUARD_ROOM_KEYWORDS):
                room = '경비실'
            else:
                try:
                    room = int(room)
                except (ValueError, TypeError):
                    room = None

        return {'room': room, 'press_bell': press}

    # ---------- 벨 위치/검증/근접 ----------
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
# 빨간색 검출기
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
# 접근 설정
# =====================================================================
APPROACH_STAGES = [
    {'target': 0.50, 'name': '50cm 정렬', 'tol': 0.10, 'confirm': 1, 'fine': False},
    {'target': 0.35, 'name': '35cm 접근', 'tol': 0.08, 'confirm': 1, 'fine': False},
    {'target': 0.25, 'name': '25cm 정밀정렬', 'tol': 0.03, 'confirm': 3, 'fine': True},
]

STOP_TOLERANCE = 0.02

FINAL_PUSH_DISTANCE = 0.08
FINAL_PUSH_SPEED = 0.03
FINAL_PUSH_TIME = FINAL_PUSH_DISTANCE / FINAL_PUSH_SPEED

LIDAR_FRONT_ANGLE = 0
LIDAR_FRONT_WINDOW_DEG = 15.0

ALIGN_STEP_TIME = 0.15
ALIGN_STEP_TIME_FINE = 0.10

SEARCH_ANGULAR = 0.35
SEARCH_STEP_TIME = 0.4
SEARCH_FULL_TURN = 2.0 * math.pi

MAX_VERIFY_ATTEMPTS = 60


# =====================================================================
# F키 대기 창 (OpenCV)
# =====================================================================
def wait_for_f_key(window_title='대기 중', message='F 키를 누르면 홈으로 복귀합니다.'):
    """OpenCV 창을 띄우고 F키 입력을 기다린다.

    Q 또는 ESC 입력 시 False(종료 요청) 반환.
    F 입력 시 True(복귀 출발) 반환.
    """
    print(f'\n[대기] {message}  (Q/ESC: 전체 종료)')

    canvas = np.zeros((200, 700, 3), dtype=np.uint8)

    # 한글은 OpenCV에서 깨지므로 영문으로 표시
    cv2.putText(canvas, 'Mission done. Press F to go home.',
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    cv2.putText(canvas, 'F: go home   Q/ESC: quit',
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

    cv2.imshow(window_title, canvas)

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('f') or key == ord('F'):
            cv2.destroyWindow(window_title)
            print('[F키] 홈 복귀 출발!')
            return True
        elif key == ord('q') or key == ord('Q') or key == 27:
            cv2.destroyWindow(window_title)
            print('[Q/ESC] 종료 요청.')
            return False


# =====================================================================
# 메인 제어 노드
# =====================================================================
class DeliveryController(Node):
    def __init__(self):
        super().__init__('delivery_controller')

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        self.camera_msg = None
        self.scan_msg = None
        self.amcl_msg = None

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

        # /amcl_pose 구독 (도착 위치 검증용)
        # VOLATILE + RELIABLE: AMCL의 latched(TRANSIENT_LOCAL) 퍼블리셔와도
        # 호환되며, 주행 중 들어오는 실시간 추정치를 받는다.
        amcl_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, amcl_qos
        )

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.detector = RedBellDetector(logger=self.get_logger())
        self.vlm = BellVLM(logger=self.get_logger())

    def _cam_cb(self, msg):
        self.camera_msg = msg

    def _scan_cb(self, msg):
        self.scan_msg = msg

    def _amcl_cb(self, msg):
        self.amcl_msg = msg

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
        return math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)

    @staticmethod
    def _quaternion_to_yaw(x, y, z, w):
        """쿼터니언 → yaw[rad] (평면 기준)."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    # =================================================================
    # 현재 위치(amcl_pose) 읽기 + 목표 오차 계산
    # =================================================================
    def _get_current_pose(self):
        """amcl_pose 기반 현재 위치 (x, y, yaw[rad]) 반환. 실패 시 None.

        주행 직후 최신 추정치를 받기 위해 잠시 spin 한다.
        AMCL 미실행 등으로 한 번도 수신하지 못하면 None.
        """
        for _ in range(AMCL_WAIT_SPINS):
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.amcl_msg is None:
            self.get_logger().warn(
                'amcl_pose 미수신 (AMCL 실행/토픽 확인 필요)'
            )
            return None

        p = self.amcl_msg.pose.pose
        yaw = self._quaternion_to_yaw(
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w
        )
        return (p.position.x, p.position.y, yaw)

    def _pose_error(self, current_pose, goal_dict):
        """현재 위치와 목표의 (거리[m], 각도[deg]) 오차를 반환."""
        cx, cy, cyaw = current_pose
        gx = float(goal_dict['x'])
        gy = float(goal_dict['y'])
        gyaw = math.radians(goal_dict.get('yaw_deg', 0.0))

        dist = math.hypot(gx - cx, gy - cy)
        yaw_err = abs(self._normalize_angle(gyaw - cyaw))
        return dist, math.degrees(yaw_err)

    # =================================================================
    # Nav2 목표 전송 + 결과 대기 (Nav2 결과 기준 성공/실패)
    # =================================================================
    def _send_nav_goal(self, pose_dict, label='목표'):
        yaw_rad = math.radians(pose_dict.get('yaw_deg', 0.0))
        qz, qw = self._yaw_to_quaternion(yaw_rad)

        pose = PoseStamped()
        pose.header.frame_id = GOAL_FRAME_ID
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(pose_dict['x'])
        pose.pose.position.y = float(pose_dict['y'])
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = pose

        self.get_logger().info(
            f'[{label}] 이동 시작: x={pose_dict["x"]}, y={pose_dict["y"]}, '
            f'yaw={pose_dict.get("yaw_deg", 0.0)}deg'
        )

        send_future = self.nav_client.send_goal_async(nav_goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Nav2가 목표를 거부했습니다.')
            return False

        self.get_logger().info(f'[{label}] Nav2 목표 수락됨. 이동 중...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        if result is None:
            self.get_logger().error('Nav2 결과 수신 실패.')
            return False

        self.get_logger().info(f'[{label}] Nav2 주행 결과 수신.')
        return True

    # =================================================================
    # Nav2 이동 + 도착 위치 검증 + 재주행 (좌표 딕셔너리 직접 수신)
    # =================================================================
    def _navigate_to_pose(self, pose_dict, label='목표'):
        if pose_dict['x'] is None or pose_dict['y'] is None:
            self.get_logger().error(f'{label} 좌표가 None입니다.')
            return False

        self.get_logger().info('Nav2 서버 대기 중...')
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 액션 서버에 연결 실패.')
            return False

        total_attempts = 1 + MAX_NAV_RETRIES
        for attempt in range(1, total_attempts + 1):
            if attempt > 1:
                self.get_logger().warn(
                    f'[{label}] 위치 오차 초과 → 재주행 시도 '
                    f'{attempt - 1}/{MAX_NAV_RETRIES}'
                )

            # 1) 실제 Nav2 주행
            nav_ok = self._send_nav_goal(pose_dict, label)
            if not nav_ok:
                self.get_logger().error(
                    f'[{label}] Nav2 주행 실패(시도 {attempt}/{total_attempts}).'
                )
                continue

            # 2) 도착 후 amcl_pose로 위치 검증
            current = self._get_current_pose()
            if current is None:
                # AMCL을 못 받으면 검증 불가 → 기존 동작처럼 도착으로 간주(안전 폴백)
                self.get_logger().warn(
                    f'[{label}] 현재 위치 확인 불가 → 검증 생략하고 도착으로 간주.'
                )
                return True

            dist_err, yaw_err = self._pose_error(current, pose_dict)
            self.get_logger().info(
                f'[{label}] 도착 검증: 거리오차={dist_err * 100:.1f}cm '
                f'(허용 {GOAL_DISTANCE_TOLERANCE * 100:.0f}cm), '
                f'각도오차={yaw_err:.1f}deg '
                f'(허용 {GOAL_YAW_TOLERANCE_DEG:.0f}deg)'
            )

            if (dist_err <= GOAL_DISTANCE_TOLERANCE
                    and yaw_err <= GOAL_YAW_TOLERANCE_DEG):
                self.get_logger().info(f'[{label}] 위치 검증 통과. 도착 완료.')
                return True

            self.get_logger().warn(
                f'[{label}] 위치 오차 허용 범위 초과.'
            )

        self.get_logger().error(
            f'[{label}] {total_attempts}회 시도했으나 목표 위치 도달 실패.'
        )
        return False

    def navigate_to_room(self, room):
        """room이 '경비실'이면 GUARD_ROOM_GOAL로, 숫자면 ROOM_GOALS[room]으로 이동."""
        if room == '경비실':
            if GUARD_ROOM_GOAL['x'] is None or GUARD_ROOM_GOAL['y'] is None:
                self.get_logger().error('경비실 좌표가 아직 입력되지 않았습니다(TODO).')
                return False
            return self._navigate_to_pose(GUARD_ROOM_GOAL, label='경비실')

        goal = ROOM_GOALS.get(room)
        if goal is None:
            self.get_logger().error(f'{room}호 좌표가 ROOM_GOALS에 없습니다.')
            return False
        if goal['x'] is None or goal['y'] is None or goal['yaw_deg'] is None:
            self.get_logger().error(
                f'{room}호 좌표/방향이 아직 입력되지 않았습니다(TODO).'
            )
            return False
        return self._navigate_to_pose(goal, label=f'{room}호')

    def navigate_to_home(self):
        self.get_logger().info('===== 홈(초기 위치)으로 복귀 시작 =====')
        result = self._navigate_to_pose(HOME_POSITION, label='홈')
        if result:
            self.get_logger().info('===== 홈 복귀 완료 =====')
        else:
            self.get_logger().error('===== 홈 복귀 실패 =====')
        return result

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
                f'{prefix}벽(벨)과의 거리: {dist*100:.1f}cm'
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
    # 빨간색 찾기 + VLM 벨 검증
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
    # 픽셀 기반 정밀 정렬
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
    # 검증 + 정렬 묶음
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
    # LiDAR 기반 목표 거리까지 전진
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
    # 버튼 누르기 전진
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
    # 벨 누르기 전체 시퀀스
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
    # 벨 누른 후 복귀 절차
    #   1) 3초 정지
    #   2) 15cm 후진
    #   3) [F키 대기] → F 입력 시 Nav2로 홈 복귀
    #
    # 반환: True(정상 완료), False(종료 요청)
    # =================================================================
    def return_after_bell(self):
        self.get_logger().info(
            f'===== 벨 눌림 후 복귀 시작 | {BELL_PRESSED_STOP_SEC:.0f}초 정지 ====='
        )
        self._stop()
        time.sleep(BELL_PRESSED_STOP_SEC)

        # 15cm 후진
        reverse_time = REVERSE_DISTANCE_M / REVERSE_SPEED
        self.get_logger().info(
            f'후진 시작: {REVERSE_DISTANCE_M*100:.0f}cm '
            f'({REVERSE_SPEED}m/s × {reverse_time:.1f}s)'
        )
        end_t = time.time() + reverse_time
        while time.time() < end_t and rclpy.ok():
            self._pub_cmd(-REVERSE_SPEED, 0.0)
            time.sleep(0.05)
        self._stop()
        self.get_logger().info('후진 완료. F키 대기 중...')
        time.sleep(0.3)

        # ★ F키 입력 대기 ★
        proceed = wait_for_f_key(
            window_title='Bell done - waiting',
            message='Bell pressed + reversed. Press F to go home.'
        )
        if not proceed:
            return False

        self.navigate_to_home()
        return True

    # =================================================================
    # 벨 안 누른 후 복귀 절차
    #   1) 제자리 180도 회전 (HALF_ROTATE_ANGULAR 속도)
    #   2) 2초 정지
    #   3) [F키 대기] → F 입력 시 Nav2로 홈 복귀
    #
    # 반환: True(정상 완료), False(종료 요청)
    # =================================================================
    def return_after_no_bell(self):
        self.get_logger().info(
            f'===== 벨 미누름 후 복귀 시작 | 180도 회전 '
            f'(속도 {HALF_ROTATE_ANGULAR}rad/s) ====='
        )

        # 180도 = π rad → 소요 시간 계산
        half_turn_time = math.pi / HALF_ROTATE_ANGULAR
        self.get_logger().info(
            f'제자리 180도 회전: {HALF_ROTATE_ANGULAR}rad/s × {half_turn_time:.1f}s'
        )
        end_t = time.time() + half_turn_time
        while time.time() < end_t and rclpy.ok():
            self._pub_cmd(0.0, HALF_ROTATE_ANGULAR)
            time.sleep(0.05)
        self._stop()
        self.get_logger().info(f'180도 회전 완료. {NO_BELL_STOP_SEC:.0f}초 정지 중...')

        # 2초 정지
        time.sleep(NO_BELL_STOP_SEC)
        self.get_logger().info('정지 완료. F키 대기 중...')

        # ★ F키 입력 대기 ★
        proceed = wait_for_f_key(
            window_title='Delivery done - waiting',
            message='Delivery done + rotated. Press F to go home.'
        )
        if not proceed:
            return False

        self.navigate_to_home()
        return True

    # =================================================================
    # 영수증 판단 결과를 받아 미션 수행 + 복귀
    #
    # 반환: (성공여부, 종료요청여부)
    # =================================================================
    def execute_mission(self, room, press_bell):
        label = '경비실' if room == '경비실' else f'{room}호'
        self.get_logger().info(
            f'===== 미션 시작: {label}, 벨 누름={press_bell} ====='
        )

        # 1) 목적지로 이동 (호수 또는 경비실)
        if not self.navigate_to_room(room):
            self.get_logger().error('이동 실패. 홈으로 복귀 시도.')
            self.navigate_to_home()
            return False, False

        # 2) 분기 + 복귀
        if press_bell:
            result = self.press_bell_sequence()
            quit_requested = not self.return_after_bell()
        else:
            self.get_logger().info('벨 누름 불필요. 도착 후 정지.')
            self._stop()
            quit_requested = not self.return_after_no_bell()
            result = True

        return result, quit_requested


# =====================================================================
# 로봇 카메라(/camera/image_raw)로 영수증 캡처 + 키 입력 UI
# =====================================================================
def capture_receipt_and_decide(node):
    """로봇 카메라 미리보기 → SPACE 캡처 → VLM 판단
       → G 출발 / R 재캡처 / Q·ESC 취소

    반환: (room, press_bell) 또는 None(취소/종료)
    room은 int(호수) 또는 str('경비실')
    """
    print("\n[로봇 카메라] 영수증을 카메라에 보여주고 SPACE로 캡처하세요. (Q/ESC: 종료)")

    decision = None
    state = 'preview'
    last_capture = None

    try:
        while rclpy.ok():
            if state == 'preview':
                frame = node._get_bgr()
                if frame is None:
                    print('로봇 카메라 프레임 수신 실패. '
                          '토픽(/camera/image_raw) 확인 중...')
                    time.sleep(0.3)
                    continue
            else:
                frame = last_capture

            view = frame.copy()
            if state == 'preview':
                cv2.putText(view, 'SPACE: capture  Q/ESC: quit',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            else:
                room_val = decision['room']
                press = decision['press_bell']
                room_display = str(room_val) if room_val == '경비실' else f'{room_val}ho'
                cv2.putText(view, f'room={room_display}  press_bell={press}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                cv2.putText(view, 'G: go  R: recapture  Q/ESC: quit',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            cv2.imshow('Receipt', view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                decision = None
                break

            if state == 'preview' and key == ord(' '):
                print('[캡처] 영수증 분석 중...')
                last_capture = frame.copy()
                result = node.vlm.read_receipt(last_capture)
                if result is None or result['room'] is None:
                    print('판단 실패(호수/경비실 인식 불가). 다시 시도하세요.')
                    continue
                decision = result
                room_str = '경비실' if decision['room'] == '경비실' else f"{decision['room']}호"
                print(f"=> 판단 결과: {room_str}, 벨 누름={decision['press_bell']}")
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


# =====================================================================
# 메인 루프
# =====================================================================
def main():
    rclpy.init()
    node = DeliveryController()

    mission_count = 0

    try:
        print("\n========================================")
        print("  배달 로봇 시작")
        print("  영수증을 인식시켜 주세요.")
        print("  (호수 또는 경비실 인식 가능)")
        print("  Q 또는 ESC 키: 전체 종료")
        print("========================================\n")

        while rclpy.ok():
            # ---- 영수증 인식 단계 ----
            decision = capture_receipt_and_decide(node)

            if decision is None:
                print('\n=== 종료 요청. 루프 종료. ===')
                break

            room, press_bell = decision
            mission_count += 1
            label = '경비실' if room == '경비실' else f'{room}호'
            print(f"\n[미션 #{mission_count}] {label}, 벨 누름={press_bell}")

            # ---- 미션 수행 ----
            result, quit_requested = node.execute_mission(room, press_bell)

            if quit_requested:
                print('\n=== 종료 요청(F키 대기 중 Q/ESC). 루프 종료. ===')
                break

            if result:
                print(f"\n=== 미션 #{mission_count} 완료. 홈 복귀 후 대기 중 ===\n")
            else:
                print(f"\n=== 미션 #{mission_count} 중단/실패. 홈 복귀 후 대기 중 ===\n")

    except KeyboardInterrupt:
        node._stop()
        print('\n긴급 정지 (Ctrl+C)')
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()
        print(f'\n총 수행 미션: {mission_count}건')


if __name__ == '__main__':
    main()
