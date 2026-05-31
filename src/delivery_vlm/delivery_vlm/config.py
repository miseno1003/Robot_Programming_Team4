"""VLM 관련 설정값과 프롬프트 모음."""

# 사용할 Claude 모델
VLM_MODEL = "claude-sonnet-4-6"
VLM_MAX_TOKENS = 256
JPEG_QUALITY = 85

# 경비실로 판단하는 키워드 (VLM 응답에 이 중 하나라도 포함되면 경비실 처리)
GUARD_ROOM_KEYWORDS = ['경비실', '경비', 'guard', 'security']

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

# --- 벨 검증 + 위치 통합 프롬프트 (API 호출 1번으로 처리) ---
BELL_VERIFY_LOCATE_PROMPT = (
    "이 이미지에 눌러야 하는 빨간색 벨(또는 둥근 빨간 버튼)이 보이나요?\n"
    "반드시 JSON 형식으로만 답하세요. 다른 설명은 붙이지 마세요.\n"
    "1) is_bell: 눌러야 하는 빨간 벨/버튼이 맞으면 true, 아니면 false.\n"
    "2) position: 벨의 화면상 위치. \"없음\" / \"왼쪽\" / \"중앙\" / \"오른쪽\" 중 하나.\n"
    '형식 예시: {"is_bell": true, "position": "중앙"}'
)
