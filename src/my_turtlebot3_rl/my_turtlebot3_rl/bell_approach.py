#!/usr/bin/env python3

import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool


# ============================================================
# Gazebo world 기준 로봇 시작 위치
# 현재 launch:
#   x_pose = -1.2
#   y_pose = -1.2
#   yaw 기본값 = 0도
# ============================================================
ROBOT_START_WORLD_X = -1.2
ROBOT_START_WORLD_Y = -1.2
ROBOT_START_WORLD_YAW = 0.0


# 벽/벨 접근 설정
APPROACH_OFFSET = 0.55

# 중앙 장애물 금지구역
# 실제 장애물은 대략 -0.5~0.5 이지만, 로봇 크기 때문에 여유를 둠
OBSTACLE_MIN_X = -0.75
OBSTACLE_MAX_X = 0.75
OBSTACLE_MIN_Y = -0.75
OBSTACLE_MAX_Y = 0.75

# 우회 waypoint 후보점
CORNER_CLEARANCE = 0.95

# 이동/접근 속도 및 안전거리
MOVE_STOP_DIST = 0.08
PRESS_STOP_DIST = 0.12

# waypoint 이동 중 버튼이 보이면 바로 카메라 추적으로 전환
EARLY_TARGET_AREA = 250.0
EARLY_TARGET_CONFIRM_COUNT = 3


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class BellApproach(Node):
    def __init__(self):
        super().__init__('bell_approach')

        # 로봇 odom 좌표
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        # 벨 world 좌표
        self.bell_world_x = None
        self.bell_world_y = None
        self.bell_world_yaw = None

        # 벨 odom 좌표
        self.bell_odom_x = None
        self.bell_odom_y = None

        # 접근 목표
        self.approach_world_x = None
        self.approach_world_y = None
        self.approach_odom_x = None
        self.approach_odom_y = None

        self.target_yaw_world = None
        self.target_yaw_odom = None
        self.wall_side = None

        # waypoint
        self.waypoints_world = []
        self.waypoints_odom = []
        self.wp_index = 0
        self.path_planned = False

        # 상태 머신
        self.stage = 'WAIT_BELL'

        # 센서 상태
        self.front_min_dist = 999.0
        self.button_pressed = False

        self.image_width = None
        self.target_seen = False
        self.target_cx = None
        self.target_area = 0.0
        self.target_seen_count = 0
        self.warned_encoding = False

        # /bell_pose QoS
        bell_qos = QoSProfile(depth=1)
        bell_qos.reliability = ReliabilityPolicy.RELIABLE
        bell_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.bell_sub = self.create_subscription(
            PoseStamped,
            '/bell_pose',
            self.bell_pose_callback,
            bell_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.button_sub = self.create_subscription(
            Bool,
            '/button_pressed',
            self.button_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('bell_approach started.')
        self.get_logger().info('Waiting for /bell_pose...')

    # ============================================================
    # 좌표 변환
    # ============================================================
    def world_to_odom_xy(self, world_x, world_y):
        dx = world_x - ROBOT_START_WORLD_X
        dy = world_y - ROBOT_START_WORLD_Y

        c = math.cos(-ROBOT_START_WORLD_YAW)
        s = math.sin(-ROBOT_START_WORLD_YAW)

        odom_x = c * dx - s * dy
        odom_y = s * dx + c * dy

        return odom_x, odom_y

    def odom_to_world_xy(self, odom_x, odom_y):
        c = math.cos(ROBOT_START_WORLD_YAW)
        s = math.sin(ROBOT_START_WORLD_YAW)

        world_x = ROBOT_START_WORLD_X + c * odom_x - s * odom_y
        world_y = ROBOT_START_WORLD_Y + s * odom_x + c * odom_y

        return world_x, world_y

    def world_yaw_to_odom_yaw(self, world_yaw):
        return normalize_angle(world_yaw - ROBOT_START_WORLD_YAW)

    # ============================================================
    # Callback
    # ============================================================
    def bell_pose_callback(self, msg):
        self.bell_world_x = msg.pose.position.x
        self.bell_world_y = msg.pose.position.y
        self.bell_world_yaw = yaw_from_quaternion(msg.pose.orientation)

        self.bell_odom_x, self.bell_odom_y = self.world_to_odom_xy(
            self.bell_world_x,
            self.bell_world_y
        )

        self.compute_approach_goal()

        self.path_planned = False
        self.wp_index = 0
        self.button_pressed = False
        self.stage = 'PLAN_PATH'

        self.get_logger().info(
            f'Received /bell_pose WORLD: '
            f'x={self.bell_world_x:.2f}, '
            f'y={self.bell_world_y:.2f}, '
            f'yaw={math.degrees(self.bell_world_yaw):.1f} deg'
        )

        self.get_logger().info(
            f'Converted bell ODOM: '
            f'x={self.bell_odom_x:.2f}, '
            f'y={self.bell_odom_y:.2f}'
        )

        self.get_logger().info(
            f'Approach WORLD=({self.approach_world_x:.2f}, {self.approach_world_y:.2f}), '
            f'ODOM=({self.approach_odom_x:.2f}, {self.approach_odom_y:.2f}), '
            f'target_yaw_odom={math.degrees(self.target_yaw_odom):.1f} deg, '
            f'wall={self.wall_side}'
        )

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        front_ranges = []

        for i, r in enumerate(ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if abs(angle) < math.radians(20.0):
                front_ranges.append(r)

        if len(front_ranges) == 0:
            self.front_min_dist = 999.0
        else:
            self.front_min_dist = float(np.min(front_ranges))

    def image_callback(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8)

            if msg.encoding == 'rgb8':
                img = img.reshape((msg.height, msg.width, 3))
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            elif msg.encoding == 'bgr8':
                bgr = img.reshape((msg.height, msg.width, 3))

            else:
                if not self.warned_encoding:
                    self.get_logger().warning(
                        f'Unsupported image encoding: {msg.encoding}'
                    )
                    self.warned_encoding = True
                return

            self.image_width = msg.width

            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 80, 80])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 80, 80])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                self.target_seen = False
                self.target_cx = None
                self.target_area = 0.0
                self.target_seen_count = 0
                return

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area < 60:
                self.target_seen = False
                self.target_cx = None
                self.target_area = 0.0
                self.target_seen_count = 0
                return

            M = cv2.moments(largest)
            if M['m00'] == 0:
                return

            cx = int(M['m10'] / M['m00'])

            self.target_seen = True
            self.target_cx = cx
            self.target_area = float(area)
            self.target_seen_count = min(self.target_seen_count + 1, 100)

        except Exception as e:
            self.get_logger().error(f'image_callback error: {e}')

    def button_callback(self, msg):
        if msg.data and not self.button_pressed:
            self.button_pressed = True
            self.stage = 'DONE'
            self.get_logger().info('Button pressed! Mission success.')

    # ============================================================
    # 목표 계산
    # ============================================================
    def compute_approach_goal(self):
        bx = self.bell_world_x
        by = self.bell_world_y

        if by > 1.3:
            self.wall_side = 'NORTH'
            self.approach_world_x = bx
            self.approach_world_y = by - APPROACH_OFFSET

        elif by < -1.3:
            self.wall_side = 'SOUTH'
            self.approach_world_x = bx
            self.approach_world_y = by + APPROACH_OFFSET

        elif bx > 1.3:
            self.wall_side = 'EAST'
            self.approach_world_x = bx - APPROACH_OFFSET
            self.approach_world_y = by

        elif bx < -1.3:
            self.wall_side = 'WEST'
            self.approach_world_x = bx + APPROACH_OFFSET
            self.approach_world_y = by

        else:
            self.wall_side = 'UNKNOWN'
            self.approach_world_x = bx
            self.approach_world_y = by

        self.approach_odom_x, self.approach_odom_y = self.world_to_odom_xy(
            self.approach_world_x,
            self.approach_world_y
        )

        self.target_yaw_world = math.atan2(
            self.bell_world_y - self.approach_world_y,
            self.bell_world_x - self.approach_world_x
        )

        self.target_yaw_odom = self.world_yaw_to_odom_yaw(
            self.target_yaw_world
        )

    # ============================================================
    # 장애물 회피 waypoint 생성
    # ============================================================
    def point_inside_obstacle(self, p):
        x, y = p
        return (
            OBSTACLE_MIN_X <= x <= OBSTACLE_MAX_X
            and OBSTACLE_MIN_Y <= y <= OBSTACLE_MAX_Y
        )

    def segment_hits_obstacle(self, p1, p2):
        """
        p1 -> p2 직선이 중앙 장애물 금지구역을 지나가는지 검사.
        샘플링 방식이라 단순하고 안정적임.
        """
        x1, y1 = p1
        x2, y2 = p2

        length = math.hypot(x2 - x1, y2 - y1)
        steps = max(2, int(length / 0.02))

        for i in range(steps + 1):
            t = i / steps
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t

            if self.point_inside_obstacle((x, y)):
                return True

        return False

    def path_is_clear(self, points):
        for i in range(len(points) - 1):
            if self.segment_hits_obstacle(points[i], points[i + 1]):
                return False
        return True

    def path_length(self, points):
        total = 0.0
        for i in range(len(points) - 1):
            total += math.hypot(
                points[i + 1][0] - points[i][0],
                points[i + 1][1] - points[i][1]
            )
        return total

    def remove_duplicate_points(self, points):
        filtered = []
        for p in points:
            if not filtered:
                filtered.append(p)
            else:
                prev = filtered[-1]
                if math.hypot(p[0] - prev[0], p[1] - prev[1]) > 0.08:
                    filtered.append(p)
        return filtered

    def plan_waypoints_world(self):
        """
        현재 로봇 위치와 벨 접근 위치를 기준으로 경로 생성.

        1. 현재 위치 -> 접근 위치 직선이 장애물과 안 만나면 바로 이동
        2. 만나면 중앙 장애물 바깥 모서리 후보점을 이용해서 우회 경로 선택
        """

        if self.robot_x is None or self.robot_y is None:
            return False

        start_world = self.odom_to_world_xy(self.robot_x, self.robot_y)
        goal_world = (self.approach_world_x, self.approach_world_y)

        corner_points = [
            (-CORNER_CLEARANCE, -CORNER_CLEARANCE),
            ( CORNER_CLEARANCE, -CORNER_CLEARANCE),
            ( CORNER_CLEARANCE,  CORNER_CLEARANCE),
            (-CORNER_CLEARANCE,  CORNER_CLEARANCE),
        ]

        candidate_paths = []

        # 1. 직선 경로
        direct_path = [start_world, goal_world]
        if self.path_is_clear(direct_path):
            candidate_paths.append(direct_path)

        # 2. 코너 1개 경유
        for c1 in corner_points:
            path = [start_world, c1, goal_world]
            if self.path_is_clear(path):
                candidate_paths.append(path)

        # 3. 코너 2개 경유
        for c1 in corner_points:
            for c2 in corner_points:
                if c1 == c2:
                    continue

                path = [start_world, c1, c2, goal_world]
                if self.path_is_clear(path):
                    candidate_paths.append(path)

        if not candidate_paths:
            self.get_logger().warning(
                'No clear path found. Falling back to direct approach.'
            )
            best_path = direct_path
        else:
            best_path = min(candidate_paths, key=self.path_length)

        # start_world는 현재 위치이므로 실제 waypoint에서는 제거
        waypoints_world = best_path[1:]
        waypoints_world = self.remove_duplicate_points(waypoints_world)

        self.waypoints_world = waypoints_world
        self.waypoints_odom = [
            self.world_to_odom_xy(wx, wy)
            for wx, wy in self.waypoints_world
        ]

        self.wp_index = 0
        self.path_planned = True

        self.get_logger().info(
            f'Robot WORLD start=({start_world[0]:.2f}, {start_world[1]:.2f})'
        )
        self.get_logger().info(
            f'Goal WORLD=({goal_world[0]:.2f}, {goal_world[1]:.2f})'
        )
        self.get_logger().info(
            f'Planned Waypoints WORLD: {self.waypoints_world}'
        )
        self.get_logger().info(
            f'Planned Waypoints ODOM: {self.waypoints_odom}'
        )

        return True

    # ============================================================
    # 제어 함수
    # ============================================================
    def publish_cmd(self, linear_x, angular_z):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = float(linear_x)
        cmd.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.publish_cmd(0.0, 0.0)

    def move_to_point(self, gx, gy):
        dx = gx - self.robot_x
        dy = gy - self.robot_y

        dist = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(target_heading - self.robot_yaw)

        angular_z = clamp(1.8 * heading_error, -0.65, 0.65)

        if abs(heading_error) > math.radians(20.0):
            linear_x = 0.0
        else:
            linear_x = clamp(0.45 * dist, 0.03, 0.13)

        if self.front_min_dist < MOVE_STOP_DIST:
            linear_x = 0.0

        self.publish_cmd(linear_x, angular_z)

        return dist, heading_error

    def align_to_yaw(self, target_yaw):
        yaw_error = normalize_angle(target_yaw - self.robot_yaw)
        angular_z = clamp(1.8 * yaw_error, -0.55, 0.55)

        if abs(yaw_error) < math.radians(3.0):
            self.stop()
            return True, yaw_error

        self.publish_cmd(0.0, angular_z)
        return False, yaw_error

    # ============================================================
    # Main loop
    # ============================================================
    def control_loop(self):
        if self.robot_x is None or self.robot_yaw is None:
            self.stop()
            return

        if self.stage == 'WAIT_BELL':
            self.stop()
            self.get_logger().info(
                'Waiting for /bell_pose...',
                throttle_duration_sec=2.0
            )
            return

        if self.button_pressed or self.stage == 'DONE':
            self.stop()
            return

        # ------------------------------------------------------------
        # 0. 현재 로봇 위치를 기준으로 경로 계획
        # ------------------------------------------------------------
        if self.stage == 'PLAN_PATH':
            if self.plan_waypoints_world():
                self.stage = 'GO_WAYPOINT'
                self.get_logger().info('Path planned. Start waypoint navigation.')
            else:
                self.stop()
            return

        # ------------------------------------------------------------
        # 1. waypoint 이동
        #    이동 중 빨간 버튼이 보이면 waypoint 생략
        # ------------------------------------------------------------
        if self.stage == 'GO_WAYPOINT':

            if (
                self.target_seen
                and self.target_seen_count >= EARLY_TARGET_CONFIRM_COUNT
                and self.target_area >= EARLY_TARGET_AREA
            ):
                self.stop()
                self.stage = 'CAMERA_ALIGN'
                self.get_logger().info(
                    f'Red button detected during waypoint navigation. '
                    f'Skip remaining waypoints. '
                    f'area={self.target_area:.1f}, '
                    f'cx={self.target_cx}, '
                    f'seen_count={self.target_seen_count}'
                )
                return

            if self.wp_index >= len(self.waypoints_odom):
                self.stage = 'ALIGN_TO_BELL'
                self.stop()
                return

            gx, gy = self.waypoints_odom[self.wp_index]
            dist, heading_error = self.move_to_point(gx, gy)

            self.get_logger().info(
                f'[GO_WAYPOINT] '
                f'wp={self.wp_index + 1}/{len(self.waypoints_odom)} '
                f'goal_odom=({gx:.2f},{gy:.2f}) '
                f'robot_odom=({self.robot_x:.2f},{self.robot_y:.2f}) '
                f'dist={dist:.2f}, '
                f'heading_err={math.degrees(heading_error):.1f} deg, '
                f'front={self.front_min_dist:.2f}',
                throttle_duration_sec=1.0
            )

            if dist < 0.08:
                self.stop()
                self.get_logger().info(
                    f'Reached waypoint {self.wp_index + 1}.'
                )
                self.wp_index += 1

                if self.wp_index >= len(self.waypoints_odom):
                    self.stage = 'ALIGN_TO_BELL'
                    self.get_logger().info(
                        'All waypoints reached. Aligning to bell.'
                    )

            return

        # ------------------------------------------------------------
        # 2. 벨 방향으로 yaw 정렬
        # ------------------------------------------------------------
        if self.stage == 'ALIGN_TO_BELL':
            aligned, yaw_error = self.align_to_yaw(self.target_yaw_odom)

            self.get_logger().info(
                f'[ALIGN_TO_BELL] '
                f'robot_yaw={math.degrees(self.robot_yaw):.1f} deg, '
                f'target_yaw={math.degrees(self.target_yaw_odom):.1f} deg, '
                f'error={math.degrees(yaw_error):.1f} deg, '
                f'front={self.front_min_dist:.2f}',
                throttle_duration_sec=1.0
            )

            if aligned:
                self.stage = 'CAMERA_ALIGN'
                self.get_logger().info(
                    'Yaw aligned. Camera alignment start.'
                )

            return

        # ------------------------------------------------------------
        # 3. 카메라로 버튼 중앙 정렬
        # ------------------------------------------------------------
        if self.stage == 'CAMERA_ALIGN':
            if not self.target_seen or self.image_width is None:
                self.publish_cmd(0.0, 0.18)
                self.get_logger().info(
                    '[CAMERA_ALIGN] red button not seen. rotating slowly.',
                    throttle_duration_sec=1.0
                )
                return

            error = (
                self.target_cx - self.image_width / 2.0
            ) / (self.image_width / 2.0)

            angular_z = clamp(-0.55 * error, -0.35, 0.35)

            self.get_logger().info(
                f'[CAMERA_ALIGN] '
                f'area={self.target_area:.1f}, '
                f'cx_error={error:.2f}, '
                f'front={self.front_min_dist:.2f}',
                throttle_duration_sec=0.5
            )

            if abs(error) < 0.08:
                self.stop()
                self.stage = 'PRESS_BUTTON'
                self.get_logger().info('Button centered. Pressing button.')
            else:
                self.publish_cmd(0.0, angular_z)

            return

        # ------------------------------------------------------------
        # 4. 버튼 누르기
        # ------------------------------------------------------------
        if self.stage == 'PRESS_BUTTON':
            if self.button_pressed:
                self.stage = 'DONE'
                self.stop()
                return

            if self.front_min_dist < PRESS_STOP_DIST:
                self.stop()
                self.get_logger().warning(
                    'Too close to wall. Stop. Button not detected.'
                )
                return

            angular_z = 0.0

            if self.target_seen and self.image_width is not None:
                error = (
                    self.target_cx - self.image_width / 2.0
                ) / (self.image_width / 2.0)

                angular_z = clamp(-0.30 * error, -0.18, 0.18)

            self.get_logger().info(
                f'[PRESS_BUTTON] '
                f'front={self.front_min_dist:.2f}, '
                f'target_seen={self.target_seen}, '
                f'area={self.target_area:.1f}',
                throttle_duration_sec=0.5
            )

            self.publish_cmd(0.06, angular_z)
            return


def main(args=None):
    rclpy.init(args=args)
    node = BellApproach()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()