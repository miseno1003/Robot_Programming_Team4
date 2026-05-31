"""BellVLM: Anthropic API 를 이용한 영수증 판단 + 벨 검증/위치 로직.

ROS2 와 무관한 순수 로직 클래스. vlm_node 에서 import 해서 사용한다.
"""

import json
import base64

import cv2
import numpy as np
import anthropic

from delivery_vlm.config import (
    VLM_MODEL,
    VLM_MAX_TOKENS,
    JPEG_QUALITY,
    GUARD_ROOM_KEYWORDS,
    RECEIPT_PROMPT,
    BELL_VERIFY_LOCATE_PROMPT,
)


class BellVLM:
    """VLM 로직: 영수증 판단 + 벨 검증/위치."""

    def __init__(self, model=VLM_MODEL, logger=None):
        # ANTHROPIC_API_KEY 환경변수를 자동으로 읽는다.
        self.client = anthropic.Anthropic()
        self.model = model
        self._logger = logger

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(msg)
        else:
            print(msg)

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

    @staticmethod
    def _extract_json(text):
        """문자열에서 첫 번째 JSON 객체를 추출해 dict 로 반환. 실패 시 None."""
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return None

    # ---------- 영수증 판단 ----------
    def read_receipt(self, bgr):
        """영수증 이미지 → (destination:str, press_bell:bool, ok:bool)

        destination: "101" 같은 호수 문자열 또는 "경비실".
                     인식 실패 시 빈 문자열 & ok=False.
        """
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return '', False, False

        answer = self.ask(b64, RECEIPT_PROMPT)
        self._log(f'VLM 영수증 응답: {answer}')

        data = self._extract_json(answer)
        if data is None:
            self._log('영수증 JSON 파싱 실패')
            return '', False, False

        room = data.get('room')
        press = bool(data.get('press_bell', False))

        if room is None:
            return '', False, False

        room_str = str(room).strip()

        # 경비실 키워드 판별
        if any(kw in room_str for kw in GUARD_ROOM_KEYWORDS):
            return '경비실', press, True

        # 숫자 호수 판별
        try:
            room_num = int(room)
            return str(room_num), press, True
        except (ValueError, TypeError):
            return '', False, False

    # ---------- 벨 검증 + 위치 ----------
    def verify_and_locate(self, bgr):
        """벨 이미지 → (is_bell:bool, position:str)

        position: "없음" | "왼쪽" | "중앙" | "오른쪽"
        """
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return False, '없음'

        answer = self.ask(b64, BELL_VERIFY_LOCATE_PROMPT)
        self._log(f'VLM 벨 검증 응답: {answer}')

        data = self._extract_json(answer)
        if data is None:
            # JSON 실패 시 텍스트 기반 폴백
            is_bell = ('예' in answer) or ('true' in answer.lower())
            if '중앙' in answer:
                pos = '중앙'
            elif '왼쪽' in answer:
                pos = '왼쪽'
            elif '오른쪽' in answer:
                pos = '오른쪽'
            else:
                pos = '없음'
            return is_bell, pos

        is_bell = bool(data.get('is_bell', False))
        pos = str(data.get('position', '없음')).strip()
        if pos not in ('없음', '왼쪽', '중앙', '오른쪽'):
            pos = '없음'
        return is_bell, pos
