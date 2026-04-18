"""
Захват жестов руки через MediaPipe Tasks API + OpenCV
Установка: pip install mediapipe opencv-python pyautogui

Управление:
  Q — выход
  1 — режим просмотра (только отображение жестов)
  2 — режим действий (жесты → клавиши/действия)

При первом запуске автоматически скачивается модель hand_landmarker.task (~8 МБ).
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import urllib.request
import os
from collections import deque

pyautogui.FAILSAFE = False

# ── Модель ───────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Скачиваем модель hand_landmarker.task…")
    import ssl, subprocess
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(MODEL_URL, context=ctx) as r, open(MODEL_PATH, "wb") as f:
            f.write(r.read())
    except Exception:
        subprocess.run(["curl", "-L", MODEL_URL, "-o", MODEL_PATH], check=True)
    print("Модель скачана.")

# ── MediaPipe Tasks setup ────────────────────────────────────────────────────

BaseOptions         = mp.tasks.BaseOptions
HandLandmarker      = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts  = mp.tasks.vision.HandLandmarkerOptions
RunningMode         = mp.tasks.vision.RunningMode

_options = HandLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6,
)
landmarker = HandLandmarker.create_from_options(_options)

# Соединения для отрисовки скелета кисти
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,17),
]

# ── Сглаживание координат ────────────────────────────────────────────────────

SMOOTH_N = 5

class Smoother:
    def __init__(self, n=SMOOTH_N):
        self.buf = deque(maxlen=n)

    def update(self, val):
        self.buf.append(val)
        return sum(self.buf) / len(self.buf)

smooth_x     = Smoother()
smooth_y     = Smoother()
smooth_pinch = Smoother(8)

# ── Индексы точек руки ───────────────────────────────────────────────────────

TIPS = [4, 8, 12, 16, 20]
PIP  = [3, 6, 10, 14, 18]

# ── Детектор жестов ──────────────────────────────────────────────────────────

MARGIN = 0.02  # минимальный зазор, чтобы не считать «на границе»

def count_fingers(lm, handedness="Right"):
    count = 0
    # Большой палец: по оси X с запасом
    if handedness == "Right":
        if lm[4].x < lm[3].x - MARGIN:
            count += 1
    else:
        if lm[4].x > lm[3].x + MARGIN:
            count += 1
    # Остальные: кончик выше среднего сустава с запасом
    for i in range(1, 5):
        if lm[TIPS[i]].y < lm[PIP[i]].y - MARGIN:
            count += 1
    return count


def get_pinch_distance(lm):
    dx = lm[4].x - lm[8].x
    dy = lm[4].y - lm[8].y
    return (dx**2 + dy**2) ** 0.5


def classify_gesture(lm, handedness="Right"):
    fingers = count_fingers(lm, handedness)
    pinch   = get_pinch_distance(lm)

    if pinch < 0.05:
        return "pinch", fingers

    idx_up  = lm[8].y  < lm[6].y
    mid_up  = lm[12].y < lm[10].y
    ring_dn = lm[16].y > lm[14].y
    pink_dn = lm[20].y > lm[18].y

    if idx_up and mid_up and ring_dn and pink_dn:
        return "victory", 2

    if fingers == 0: return "fist",  0
    if fingers == 5: return "open",  5
    if fingers == 1: return "point", 1

    return f"{fingers}_fingers", fingers

# ── Дебаунс ──────────────────────────────────────────────────────────────────

class Debounce:
    def __init__(self, cooldown=0.8):
        self.cooldown = cooldown
        self.last = {}

    def ready(self, key):
        now = time.time()
        if now - self.last.get(key, 0) >= self.cooldown:
            self.last[key] = now
            return True
        return False

debounce = Debounce(cooldown=0.8)


class GestureStabilizer:
    """Подтверждает жест только если он держится CONFIRM кадров подряд."""
    CONFIRM = 6

    def __init__(self):
        self._candidate = None
        self._count     = 0
        self._stable    = "—"

    def update(self, raw: str) -> str:
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate = raw
            self._count     = 1
        if self._count >= self.CONFIRM:
            self._stable = self._candidate
        return self._stable

stabilizer = GestureStabilizer()

# ── Действия по жестам ───────────────────────────────────────────────────────

GESTURE_LABELS = {
    "fist":      "Fist",
    "open":      "Open hand",
    "point":     "Point",
    "victory":   "Victory",
    "pinch":     "Pinch",
    "2_fingers": "2 fingers",
    "3_fingers": "3 fingers",
    "4_fingers": "4 fingers",
}

GESTURE_ACTIONS = {
    "fist":    "Space",
    "point":   "-> next",
    "victory": "<- prev",
    "open":    "Vol +",
    "pinch":   "Vol -",
}

def handle_action(gesture):
    if gesture == "fist" and debounce.ready("fist"):
        print("-> space")
        pyautogui.press("space")
    elif gesture == "point" and debounce.ready("point"):
        print("-> right")
        pyautogui.press("right")
    elif gesture == "victory" and debounce.ready("victory"):
        print("-> left")
        pyautogui.press("left")
    elif gesture == "open" and debounce.ready("open"):
        print("-> volume up")
        pyautogui.press("volumeup")
    elif gesture == "pinch" and debounce.ready("pinch"):
        print("-> volume down")
        pyautogui.press("volumedown")

# ── Отрисовка скелета руки ───────────────────────────────────────────────────

def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (100, 220, 100), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 5 if i in TIPS else 3
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), r, (80, 160, 80), 1)

# ── Боковая панель ───────────────────────────────────────────────────────────

PANEL_W   = 320
BG_DARK   = (18, 18, 24)
BG_CARD   = (30, 30, 40)
ACCENT    = (167, 139, 250)   # фиолетовый
GREEN     = (80, 200, 120)
YELLOW    = (60, 200, 200)
GRAY      = (100, 100, 120)
WHITE     = (230, 230, 240)

import numpy as np

def _text(img, txt, pos, scale, color, thickness=1):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def _rect(img, pt1, pt2, color, alpha=1.0):
    """Закрашенный прямоугольник с опциональной прозрачностью."""
    if alpha < 1.0:
        ov = img.copy()
        cv2.rectangle(ov, pt1, pt2, color, -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    else:
        cv2.rectangle(img, pt1, pt2, color, -1)

def draw_finger_dots(panel, fingers, cx, cy):
    """Пять кружков — закрашены столько, сколько пальцев."""
    r, gap = 10, 26
    total_w = 5 * r * 2 + 4 * (gap - 2 * r)
    x0 = cx - total_w // 2
    for i in range(5):
        x = x0 + i * gap
        active = (i < fingers)
        color  = ACCENT if active else (50, 50, 65)
        cv2.circle(panel, (x + r, cy), r, color, -1)
        cv2.circle(panel, (x + r, cy), r, (80, 80, 100), 1)

def build_panel(h, gesture, fingers, mode, fps):
    panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
    panel[:] = BG_DARK
    W = PANEL_W
    y = 0

    # ── Заголовок ──
    _rect(panel, (0, 0), (W, 54), BG_CARD)
    _text(panel, "Hand Gesture", (16, 22), 0.55, ACCENT, 1)
    _text(panel, "Control", (16, 44), 0.55, ACCENT, 1)

    # ── FPS + Режим ──
    mode_color = GREEN if mode == 2 else YELLOW
    mode_txt   = "ACTIONS" if mode == 2 else "VIEW"
    _rect(panel, (0, 58), (W, 90), BG_CARD)
    _text(panel, f"FPS {fps:.0f}", (16, 80), 0.5, GRAY, 1)
    _text(panel, mode_txt, (W - 110, 80), 0.5, mode_color, 1)

    # ── Текущий жест ──
    y = 102
    _rect(panel, (12, y), (W - 12, y + 90), BG_CARD)
    _text(panel, "GESTURE", (24, y + 22), 0.42, GRAY)
    label = GESTURE_LABELS.get(gesture, gesture)
    # Подбираем масштаб чтобы текст влез
    scale = 0.9 if len(label) <= 10 else 0.68
    _text(panel, label, (24, y + 60), scale, WHITE, 2)

    # Точки пальцев
    draw_finger_dots(panel, fingers, W // 2, y + 80)

    # ── Разделитель ──
    y = 208
    cv2.line(panel, (16, y), (W - 16, y), (40, 40, 55), 1)

    # ── Список жестов ──
    y = 220
    _text(panel, "GESTURES", (16, y), 0.42, GRAY)
    y += 16

    gesture_order = ["fist", "point", "victory", "open", "pinch",
                     "2_fingers", "3_fingers", "4_fingers"]
    for key in gesture_order:
        lbl = GESTURE_LABELS[key]
        act = GESTURE_ACTIONS.get(key, "")
        is_active = (key == gesture)
        row_y = y + 4

        if is_active:
            _rect(panel, (8, row_y - 2), (W - 8, row_y + 26), ACCENT, alpha=0.18)
            cv2.rectangle(panel, (8, row_y - 2), (W - 8, row_y + 26), ACCENT, 1)

        name_color = WHITE if is_active else (160, 160, 180)
        act_color  = ACCENT if is_active else GRAY
        _text(panel, lbl, (18, row_y + 16), 0.46, name_color)
        if act:
            tw = cv2.getTextSize(act, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
            _text(panel, act, (W - tw - 12, row_y + 14), 0.38, act_color)

        y += 32

    # ── Подсказка управления ──
    cv2.line(panel, (16, h - 50), (W - 16, h - 50), (40, 40, 55), 1)
    _text(panel, "1=view  2=actions  Q=quit", (12, h - 16), 0.38, GRAY)

    return panel


def build_frame(cam_frame, gesture, fingers, mode, fps, hand_x, hand_y):
    """Объединяет кадр камеры и боковую панель в одно окно."""
    h, w = cam_frame.shape[:2]

    # Точка центра ладони на кадре
    if hand_x or hand_y:
        px, py = int(hand_x * w), int(hand_y * h)
        cv2.circle(cam_frame, (px, py), 12, ACCENT, 2)
        cv2.circle(cam_frame, (px, py), 3,  ACCENT, -1)

    panel = build_panel(h, gesture, fingers, mode, fps)
    return np.hstack([cam_frame, panel])

# ── Главный цикл ─────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mode      = 1
    prev_time = time.time()
    start_ms  = int(time.time() * 1000)

    print("Запуск… нажми Q для выхода, 1/2 для переключения режима")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        timestamp_ms = int(now * 1000) - start_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        fingers = 0
        hand_x = hand_y = 0.0

        if result.hand_landmarks:
            lm         = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name

            palm_pts = [0, 5, 9, 13, 17]
            raw_x = sum(lm[i].x for i in palm_pts) / len(palm_pts)
            raw_y = sum(lm[i].y for i in palm_pts) / len(palm_pts)
            hand_x = smooth_x.update(raw_x)
            hand_y = smooth_y.update(raw_y)

            raw_gesture, fingers = classify_gesture(lm, handedness)
            gesture = stabilizer.update(raw_gesture)

            if mode == 2:
                handle_action(gesture)

            draw_hand(frame, lm)
        else:
            gesture = stabilizer.update("—")

        canvas = build_frame(frame, gesture, fingers, mode, fps, hand_x, hand_y)
        cv2.imshow("Hand Gestures", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            mode = 1
            print("Режим: просмотр")
        elif key == ord('2'):
            mode = 2
            print("Режим: действия (жесты → клавиши)")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
