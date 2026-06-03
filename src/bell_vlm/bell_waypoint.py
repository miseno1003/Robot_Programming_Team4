#!/usr/bin/env python3
"""벨 단계적 접근 내비게이터

흐름:
  1. Nav2 분할 이동
     - 현재 위치(TF map->base_link)에서 벨 정면 50cm 지점까지의 직선을
       0.5m 간격 웨이포인트로 분할하여 하나씩 NavigateToPose로 이동.
  2. 초기 정렬
     - 벨 50cm 앞에서 VLM으로 벨을 화면 중앙에 정렬.
  3. 전진 + 정렬 반복
     - LiDAR 정면 거리를 기준으로 10cm씩 전진(다음 목표 거리까지).
     - 매 10cm 전진 후 VLM 정렬.
     - 50 -> 40 -> 30 -> 25cm 순으로 접근.
     - LiDAR 측정 실패 시 (속도 x 시간) 기반 폴백으로 10cm 추정 전진.
  4. 최종 정렬
     - 25cm 지점에서 VLM 최종 정렬.
  5. 최종 전진
     - 10cm 전진하여 15cm 지점(버튼 접촉 위치) 도달.

LiDAR 방향 주의:
  - 로봇 장착 상태에 따라 LaserScan의 0 rad가 정면이 아닐 수 있음.
  - 실제 정면 기준각을 LIDAR_FRONT_ANGLE 하나로 통일 관리한다.
    0.0 -> scan 0 rad가 정면, math.pi -> scan 180도가 정면.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, TwistStamped
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

import math
import time
import anthropic
import base64
import cv2
import numpy as np


# ===== 벨 설정 =====
BELL_FRONT_X = 1.5393585595032155
BELL_FRONT_Y = 0.3664660158845444
BELL_ORIENT_Z = 0.7031942508276131
BELL_ORIENT_W = 0.7109977817145368

# 로봇이 벨을 바라볼 때의 yaw (모든 웨이포인트 orientation에 사용)
YAW = 2.0 * math.atan2(BELL_ORIENT_Z, BELL_ORIENT_W)

# ===== 단계적 접근 거리 설정 (단위: m) =====
START_DISTANCE = 0.50    # Nav2로 도달할 벨 정면 거리 (50cm)
STEP_DISTANCE = 0.10     # 한 번에 줄이는 거리 (10cm)
FINAL_ALIGN_DISTANCE = 0.25   # 최종 정렬을 수행할 거리 (25cm)
CONTACT_DISTANCE = 0.15  # 최종 접촉 거리 (15cm) = 25cm 정렬 후 10cm 전진

# Nav2 웨이포인트 분할 간격 (m)
WAYPOINT_SPACING = 0.50

# ===== 거리 판정 허용 오차 =====
STOP_TOLERANCE = 0.02    # 목표 거리 도달 판정 오차 (2cm)

# ===== 전진 속도 / 시간 폴백 설정 =====
APPROACH_SPEED = 0.05    # 전진 기본 속도 (m/s)
SLOW_SPEED = 0.03        # 목표 근접 시 감속 속도 (m/s)

# LiDAR 측정 실패 시 시간 기반 폴백: 10cm = 0.05m/s x 2.0s
FALLBACK_SPEED = 0.05
FALLBACK_TIME = STEP_DISTANCE / FALLBACK_SPEED  # = 2.0s

# 최종 10cm 전진도 동일 속도/시간 폴백 사용
FINAL_PUSH_SPEED = 0.05
FINAL_PUSH_TIME = STEP_DISTANCE / FINAL_PUSH_SPEED  # = 2.0s

# ===== LiDAR 방향 보정 =====
# 실제 로봇 정면 기준각. 장착 상태에 맞춰 0.0 또는 math.pi 로 설정.
LIDAR_FRONT_ANGLE = 0.0
LIDAR_FRONT_WINDOW_DEG = 15.0   # 정면 판정 각도 범위 (+- 도)


print(f"벨 정면 기준 좌표: ({BELL_FRONT_X}, {BELL_FRONT_Y})")
print(f"로봇 목표 yaw: {math.degrees(YAW):.1f}도")
print(f"Nav2 도달 거리(START): {START_DISTANCE * 100:.0f}cm")
print(f"단계 전진 거리(STEP): {STEP_DISTANCE * 100:.0f}cm")
print(f"최종 정렬 거리: {FINAL_ALIGN_DISTANCE * 100:.0f}cm")
print(f"최종 접촉 거리(CONTACT): {CONTACT_DISTANCE * 100:.0f}cm")
print(f"웨이포인트 간격: {WAYPOINT_SPACING * 100:.0f}cm")
print(f"LiDAR 정면 기준각: {math.degrees(LIDAR_FRONT_ANGLE):.1f}도, "
      f"범위 +-{LIDAR_FRONT_WINDOW_DEG:.1f}도")
print(f"시간 폴백: {FALLBACK_SPEED}m/s x {FALLBACK_TIME:.1f}s = "
      f"{FALLBACK_SPEED * FALLBACK_TIME * 100:.0f}cm")


class BellNavigator(Node):
    def __init__(self):
        super().__init__('bell_navigator')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        self.camera_msg = None
        self.scan_msg = None

        self.cam_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._cam_cb,
            1
        )

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_cb,
            scan_qos
        )

        # 현재 위치를 읽기 위한 TF 리스너 (map -> base_link)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.vlm_client = anthropic.Anthropic()

    def _cam_cb(self, msg):
        self.camera_msg = msg

    def _scan_cb(self, msg):
        self.scan_msg = msg

    # ===== cmd_vel 헬퍼 =====
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
        time.sleep(0.3)

    # ===== 각도 처리 헬퍼 =====
    def _normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _angle_diff(self, a, b):
        return abs(self._normalize_angle(a - b))

    # ===== 현재 로봇 위치 (TF map->base_link) =====
    def _get_robot_pose(self, timeout_sec=5.0):
        """map 좌표계 기준 현재 (x, y) 반환. 실패 시 None."""
        end_time = time.time() + timeout_sec

        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                tf = self.tf_buffer.lookup_transform(
                    'map',
                    'base_link',
                    rclpy.time.Time()
                )
                x = tf.transform.translation.x
                y = tf.transform.translation.y
                return (x, y)
            except (LookupException, ConnectivityException,
                    ExtrapolationException):
                continue

        self.get_logger().warn('TF map->base_link 조회 실패')
        return None

    # ===== LiDAR 거리 측정 =====
    def _get_front_distance(self):
        """실제 로봇 정면(LIDAR_FRONT_ANGLE 기준 +-window)의 최소 거리 반환."""
        self.scan_msg = None

        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.scan_msg is not None:
                break

        if self.scan_msg is None:
            self.get_logger().warn('scan_msg 수신 실패')
            return None

        scan = self.scan_msg

        if len(scan.ranges) == 0:
            self.get_logger().warn('scan.ranges가 비어 있음')
            return None

        if scan.angle_increment == 0.0:
            self.get_logger().warn('scan.angle_increment가 0임')
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
                f'정면(기준각 {math.degrees(target_angle):.0f}도 '
                f'+-{LIDAR_FRONT_WINDOW_DEG:.0f}도) 범위에 유효 거리 없음'
            )
            return None

        front_dist = min(valid)

        self.get_logger().info(
            f'LiDAR 정면 거리: {front_dist:.3f}m '
            f'({front_dist * 100:.1f}cm)'
        )

        return front_dist

    # ===== 카메라 + VLM =====
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
            msg.height,
            msg.width,
            -1
        )

        if msg.encoding == 'rgb8':
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            bgr = raw
        else:
            bgr = raw

        success, jpeg = cv2.imencode(
            '.jpg',
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if not success:
            return None

        return base64.b64encode(jpeg.tobytes()).decode('utf-8')

    def _ask_vlm(self, b64, question):
        response = self.vlm_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
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
                        {
                            "type": "text",
                            "text": question
                        },
                    ],
                }
            ],
        )

        return response.content[0].text

    # ===== Nav2 컨트롤러 lifecycle =====
    def _deactivate_controller(self):
        self.get_logger().info('Nav2 controller_server 비활성화 중...')

        client = self.create_client(
            ChangeState,
            '/controller_server/change_state'
        )

        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('controller_server 서비스 없음, 계속 진행')
            return

        req = ChangeState.Request()
        req.transition = Transition()
        req.transition.id = Transition.TRANSITION_DEACTIVATE

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info('controller_server 비활성화 성공')
        else:
            self.get_logger().warn('controller_server 비활성화 실패 또는 응답 없음')

        time.sleep(0.5)

    def _reactivate_controller(self):
        self.get_logger().info('Nav2 controller_server 다시 활성화 시도...')

        client = self.create_client(
            ChangeState,
            '/controller_server/change_state'
        )

        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('controller_server 서비스 없음')
            return

        req = ChangeState.Request()
        req.transition = Transition()
        req.transition.id = Transition.TRANSITION_ACTIVATE

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info('controller_server 다시 활성화 성공')
        else:
            self.get_logger().warn('controller_server 다시 활성화 실패 또는 응답 없음')

    # ===== 단일 웨이포인트 전송 =====
    def _send_waypoint(self, x, y, label=''):
        """단일 NavigateToPose 목표를 보내고 성공 여부 반환."""
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0

        # 모든 웨이포인트는 벨을 향하도록 동일 orientation 사용
        goal.pose.pose.orientation.z = BELL_ORIENT_Z
        goal.pose.pose.orientation.w = BELL_ORIENT_W

        self.get_logger().info(
            f'웨이포인트 전송 {label}: x={x:.3f}, y={y:.3f}'
        )

        send_goal_future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()

        if goal_handle is None:
            self.get_logger().error('goal handle 수신 실패')
            return False

        if not goal_handle.accepted:
            self.get_logger().error('웨이포인트 목표 거부됨')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()

        if result is None:
            self.get_logger().error('결과 수신 실패')
            return False

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'웨이포인트 {label} 도착')
            return True

        self.get_logger().error(
            f'웨이포인트 {label} 실패. status={result.status}'
        )
        return False

    # ===== 1단계: Nav2 분할 이동 =====
    def go_to_bell_in_steps(self):
        self.get_logger().info('===== 1단계: Nav2 분할 이동 =====')
        self.get_logger().info('navigate_to_pose action server 대기 중...')

        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('navigate_to_pose action server 연결 실패')
            return False

        # 최종 목표: 벨 정면 START_DISTANCE(50cm) 지점
        goal_x = BELL_FRONT_X - START_DISTANCE * math.cos(YAW)
        goal_y = BELL_FRONT_Y - START_DISTANCE * math.sin(YAW)

        # 현재 로봇 위치
        start = self._get_robot_pose()
        if start is None:
            self.get_logger().warn(
                '현재 위치 조회 실패. 분할 없이 최종 지점으로 한 번에 이동합니다.'
            )
            return self._send_waypoint(goal_x, goal_y, label='(최종 50cm)')

        sx, sy = start
        dx = goal_x - sx
        dy = goal_y - sy
        total_dist = math.hypot(dx, dy)

        self.get_logger().info(
            f'시작: ({sx:.3f}, {sy:.3f}) -> '
            f'목표(50cm 지점): ({goal_x:.3f}, {goal_y:.3f}), '
            f'직선 거리: {total_dist:.3f}m'
        )

        # 웨이포인트 개수 계산 (간격 단위로 분할, 마지막은 항상 최종 지점)
        if total_dist <= WAYPOINT_SPACING:
            num_segments = 1
        else:
            num_segments = math.ceil(total_dist / WAYPOINT_SPACING)

        self.get_logger().info(
            f'웨이포인트 {num_segments}개로 분할 (간격 {WAYPOINT_SPACING*100:.0f}cm)'
        )

        for k in range(1, num_segments + 1):
            ratio = k / num_segments
            wx = sx + dx * ratio
            wy = sy + dy * ratio

            ok = self._send_waypoint(
                wx, wy, label=f'{k}/{num_segments}'
            )
            self._stop()

            if not ok:
                self.get_logger().error(
                    f'웨이포인트 {k}/{num_segments}에서 이동 실패. 중단.'
                )
                return False

        self.get_logger().info('Nav2 분할 이동 완료. 벨 50cm 앞 도착.')

        # 이후 cmd_vel 직접 제어를 위해 컨트롤러 비활성화
        self._deactivate_controller()
        return True

    # ===== VLM 정렬 (1회 호출 = 한 번 중앙 정렬까지) =====
    def align_to_bell(self, tag=''):
        self.get_logger().info(f'===== VLM 정렬 시작 {tag} =====')

        attempt = 0

        while rclpy.ok():
            attempt += 1
            self.get_logger().info(f'정렬 시도 {attempt} {tag}')

            dist = self._get_front_distance()
            if dist is not None:
                self.get_logger().info(
                    f'현재 LiDAR 거리: {dist * 100:.1f}cm'
                )

            b64 = self._get_image_b64()

            if b64 is None:
                self.get_logger().error('카메라 이미지 수신 실패. 재시도...')
                time.sleep(1.0)
                continue

            answer = self._ask_vlm(
                b64,
                "이 이미지에 빨간색 벨 또는 둥근 빨간 버튼이 보이나요?\n"
                "반드시 다음 형식 중 하나로만 답하세요:\n"
                "1) 안 보임\n"
                "2) 보임-왼쪽\n"
                "3) 보임-중앙\n"
                "4) 보임-오른쪽\n"
                "번호와 해당 텍스트만 답하세요."
            )

            self.get_logger().info(f'VLM 응답: {answer}')

            if '중앙' in answer or '3)' in answer or '3' == answer.strip():
                self._stop()
                self.get_logger().info(f'벨 중앙 정렬 완료 {tag}')
                return True

            elif '왼쪽' in answer or '2)' in answer or '2' == answer.strip():
                self.get_logger().info('벨 왼쪽 -> 반시계 회전')
                self._rotate(0.3, 0.5)

            elif '오른쪽' in answer or '4)' in answer or '4' == answer.strip():
                self.get_logger().info('벨 오른쪽 -> 시계 회전')
                self._rotate(-0.3, 0.5)

            else:
                self.get_logger().info('벨 안 보임 -> 탐색 회전')
                self._rotate(0.4, 0.8)

            time.sleep(0.5)

        self._stop()
        return False

    # ===== 시간 기반 폴백 전진 =====
    def _advance_by_time(self, speed, duration):
        """LiDAR 실패 시 (속도 x 시간)으로 거리 추정 전진."""
        self.get_logger().warn(
            f'시간 기반 폴백 전진: {speed:.3f}m/s x {duration:.2f}s '
            f'= {speed * duration * 100:.0f}cm'
        )
        end_time = time.time() + duration
        while time.time() < end_time and rclpy.ok():
            self._pub_cmd(speed, 0.0)
            time.sleep(0.05)
        self._stop()
        time.sleep(0.3)

    # ===== LiDAR 기준 목표 거리까지 전진 (실패 시 시간 폴백) =====
    def _advance_to_distance(self, target_dist):
        """현재 위치에서 LiDAR 정면 거리가 target_dist가 될 때까지 전진.

        LiDAR 측정이 연속 실패하면 시간 기반 폴백으로 STEP_DISTANCE만큼 추정 전진.
        """
        self.get_logger().info(
            f'목표 거리 {target_dist * 100:.0f}cm까지 전진'
        )

        consecutive_fail = 0
        MAX_FAIL = 5

        while rclpy.ok():
            dist = self._get_front_distance()

            if dist is None:
                consecutive_fail += 1
                self.get_logger().warn(
                    f'LiDAR 측정 실패 ({consecutive_fail}/{MAX_FAIL})'
                )
                self._stop()

                if consecutive_fail >= MAX_FAIL:
                    # LiDAR 연속 실패 -> 시간 기반으로 10cm 추정 전진
                    self._advance_by_time(FALLBACK_SPEED, FALLBACK_TIME)
                    return False  # 시간 폴백으로 처리됨

                time.sleep(0.3)
                continue

            consecutive_fail = 0
            remaining = dist - target_dist

            self.get_logger().info(
                f'현재 {dist*100:.1f}cm | 목표 {target_dist*100:.0f}cm | '
                f'남음 {remaining*100:.1f}cm'
            )

            if remaining <= STOP_TOLERANCE:
                self._stop()
                self.get_logger().info(
                    f'목표 거리 {target_dist*100:.0f}cm 도달 '
                    f'(현재 {dist*100:.1f}cm)'
                )
                return True

            # 남은 거리에 따라 속도 조절
            if remaining > 0.10:
                speed = APPROACH_SPEED
            else:
                speed = SLOW_SPEED

            self._pub_cmd(speed, 0.0)
            time.sleep(0.1)

        self._stop()
        return False

    # ===== 2~4단계: 전진 + 정렬 반복 =====
    def step_approach_with_alignment(self):
        """50cm에서 시작해 10cm씩 전진/정렬 반복, 25cm 최종 정렬까지."""
        self.get_logger().info('===== 전진 + 정렬 반복 단계 =====')

        # 50cm 지점에서 먼저 초기 정렬
        if not self.align_to_bell(tag='(초기 50cm)'):
            self.get_logger().error('초기 정렬 실패')
            return False

        # 현재 거리 측정 (없으면 START_DISTANCE로 가정)
        current = self._get_front_distance()
        if current is None:
            current = START_DISTANCE
            self.get_logger().warn(
                f'현재 거리 측정 실패. {START_DISTANCE*100:.0f}cm로 가정'
            )

        # 25cm까지 10cm 단위로 줄여가며 전진/정렬
        # 목표 거리 시퀀스 생성: 현재 -10 -> -10 ... -> 25cm 직전까지
        target = current - STEP_DISTANCE

        while target >= FINAL_ALIGN_DISTANCE - STOP_TOLERANCE and rclpy.ok():
            self.get_logger().info(
                f'--- 다음 목표 거리: {target*100:.0f}cm ---'
            )

            self._advance_to_distance(target)

            # 전진 후 정렬
            if not self.align_to_bell(tag=f'({target*100:.0f}cm)'):
                self.get_logger().error('정렬 실패. 중단')
                return False

            # 다음 목표 갱신: 현재 측정값 기준으로 다시 10cm
            now = self._get_front_distance()
            if now is None:
                now = target  # 측정 실패 시 목표값을 현재로 간주
            target = now - STEP_DISTANCE

        # ===== 4단계: 25cm 최종 정렬 =====
        self.get_logger().info('===== 최종 정렬 (25cm) =====')

        # 25cm에 정확히 맞추기 위한 추가 전진 (이미 25cm 근처면 바로 통과)
        self._advance_to_distance(FINAL_ALIGN_DISTANCE)

        if not self.align_to_bell(tag='(최종 25cm)'):
            self.get_logger().error('최종 정렬 실패')
            return False

        return True

    # ===== 5단계: 최종 10cm 전진 (25 -> 15cm) =====
    def final_push(self):
        self.get_logger().info('===== 최종 전진 (25cm -> 15cm) =====')

        # LiDAR 기준으로 15cm까지 전진, 실패 시 시간 폴백
        ok = self._advance_to_distance(CONTACT_DISTANCE)

        if not ok:
            # _advance_to_distance 내부에서 시간 폴백이 일어났거나 미도달
            self.get_logger().warn(
                'LiDAR 기준 15cm 도달 실패. 시간 기반 최종 전진 보강.'
            )
            self._advance_by_time(FINAL_PUSH_SPEED, FINAL_PUSH_TIME)

        self._stop()
        self.get_logger().info('최종 전진 완료. 버튼 접촉 시도 지점(15cm) 도달.')

        # 최종 VLM 확인
        b64 = self._get_image_b64()
        if b64:
            answer = self._ask_vlm(
                b64,
                "빨간색 벨 또는 둥근 빨간 버튼이 바로 앞에 가까이 보이나요? "
                "간단히 답해주세요."
            )
            final_dist = self._get_front_distance()

            print("\n=== 최종 상태 ===")
            if final_dist is not None:
                print(f"LiDAR 정면 거리: {final_dist * 100:.1f}cm")
            else:
                print("LiDAR 정면 거리: 측정 실패")
            print(f"VLM 판단: {answer}")


def main():
    rclpy.init()
    node = BellNavigator()

    nav2_success = False

    try:
        # 1단계: Nav2 분할 이동 (50cm 지점까지)
        nav2_success = node.go_to_bell_in_steps()

        if not nav2_success:
            node.get_logger().error('Nav2 분할 이동 실패. 이후 단계 중단.')
            return

        # 2~4단계: 전진 + 정렬 반복 -> 25cm 최종 정렬
        if not node.step_approach_with_alignment():
            node.get_logger().error('전진/정렬 단계 실패. 중단.')
            return

        # 5단계: 최종 10cm 전진 (15cm 접촉 지점)
        node.final_push()

        print("\n=== 전체 미션 완료 ===")

    except KeyboardInterrupt:
        node._stop()
        print('\n긴급 정지')

    finally:
        node._stop()

        if nav2_success:
            node._reactivate_controller()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
