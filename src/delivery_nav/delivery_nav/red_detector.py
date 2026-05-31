"""RedBellDetector: HSV 기반 빨간색 검출. ROS2 와 무관한 순수 OpenCV 로직."""

import cv2
import numpy as np

from delivery_nav.config import (
    RED_LOWER1, RED_UPPER1, RED_LOWER2, RED_UPPER2,
    MIN_RED_AREA, POSITION_CENTER_BAND,
)


class RedBellDetector:
    def __init__(self, logger=None):
        self._logger = logger

    def detect(self, bgr):
        """반환: (found, error, area, debug)
        error: 화면 중앙 기준 좌우 오차 (-1.0 ~ +1.0)
        debug: (cx, cy, width) 또는 None
        """
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
