#!/usr/bin/env python3
"""VLM 단독 테스트 — 카메라 이미지 한 장 → Claude API → 응답 출력"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import anthropic
import base64

# 1) 이미지 한 장 받기
rclpy.init()
node = Node('vlm_test')
msg = None

def cb(m):
    global msg
    msg = m

sub = node.create_subscription(Image, '/camera/image_raw', cb, 1)
print("카메라 이미지 대기 중...")

while msg is None:
    rclpy.spin_once(node, timeout_sec=1.0)

print(f"이미지 수신 완료: {msg.width}x{msg.height}, 인코딩: {msg.encoding}")
node.destroy_node()
rclpy.shutdown()

# 2) raw 이미지 → JPEG → base64
import cv2
import numpy as np

# encoding에 따라 채널 변환
raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
if msg.encoding == 'rgb8':
    bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
elif msg.encoding == 'bgr8':
    bgr = raw
else:
    bgr = raw  # 일단 그대로

_, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
print(f"JPEG 인코딩 완료: {len(jpeg)} bytes")

# 3) Claude API 호출
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
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
            {
                "type": "text",
                "text": "이 이미지에 무엇이 보이는지 한국어로 설명해줘.",
            },
        ],
    }],
)

print("\n=== Claude 응답 ===")
print(response.content[0].text)
