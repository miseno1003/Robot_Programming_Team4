"""OpenCV 기반 사용자 인터페이스: 영수증 캡처 + F키 대기 창.

ROS2 와 직접 결합하지 않도록, 프레임 획득/분석 동작은 콜백 함수로 주입받는다.
"""

import time
import cv2
import numpy as np


def wait_for_f_key(window_title='대기 중', message='Press F to go home.'):
    """OpenCV 창을 띄우고 F키 입력 대기.
    F → True(복귀 출발), Q/ESC → False(종료 요청)."""
    print(f'\n[대기] {message}  (Q/ESC: 전체 종료)')

    canvas = np.zeros((200, 700, 3), dtype=np.uint8)
    cv2.putText(canvas, 'Mission done. Press F to go home.',
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, 'F: go home   Q/ESC: quit',
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow(window_title, canvas)

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key in (ord('f'), ord('F')):
            cv2.destroyWindow(window_title)
            print('[F키] 홈 복귀 출발!')
            return True
        elif key in (ord('q'), ord('Q'), 27):
            cv2.destroyWindow(window_title)
            print('[Q/ESC] 종료 요청.')
            return False


def capture_receipt_and_decide(get_frame_fn, analyze_fn):
    """로봇 카메라 미리보기 → SPACE 캡처 → analyze_fn 호출
       → G 출발 / R 재캡처 / Q·ESC 취소.

    get_frame_fn() -> bgr(np.ndarray) or None
    analyze_fn()   -> (success, destination, press_bell)
                      (현재 카메라 프레임을 노드가 직접 캡처해 서비스 호출)

    반환: (destination:str, press_bell:bool) 또는 None(취소).
    """
    print("\n[로봇 카메라] 영수증을 보여주고 SPACE로 캡처하세요. (Q/ESC: 종료)")

    decision = None       # (destination, press_bell)
    state = 'preview'
    last_capture = None

    try:
        while True:
            if state == 'preview':
                frame = get_frame_fn()
                if frame is None:
                    print('카메라 프레임 수신 실패. (/camera/image_raw 확인 중...)')
                    time.sleep(0.3)
                    continue
            else:
                frame = last_capture

            view = frame.copy()
            if state == 'preview':
                cv2.putText(view, 'SPACE: capture  Q/ESC: quit',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                dest, press = decision
                cv2.putText(view, f'dest={dest}  press_bell={press}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(view, 'G: go  R: recapture  Q/ESC: quit',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Receipt', view)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                decision = None
                break

            if state == 'preview' and key == ord(' '):
                print('[캡처] 영수증 분석 중...')
                last_capture = frame.copy()
                success, dest, press = analyze_fn()
                if not success or not dest:
                    print('판단 실패(호수/경비실 인식 불가). 다시 시도하세요.')
                    continue
                decision = (dest, press)
                print(f"=> 판단 결과: {dest}, 벨 누름={press}")
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

    return decision
