"""
Gesture Viewer — shows which hand gesture you're making in real time.
Uses the same sliding-window stabilizer as the game: a gesture must appear
in N of the last WINDOW frames before it's confirmed.

Press Q to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import subprocess
from collections import deque

# ── MediaPipe ─────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    subprocess.run(["curl", "-L", MODEL_URL, "-o", MODEL_PATH], check=True)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
)

TIPS   = [4, 8, 12, 16, 20]
PIP    = [3, 6, 10, 14, 18]
MARGIN = 0.02

# ── Layout ────────────────────────────────────────────────────────────────────

W, H   = 1280, 720
C_DARK = (10, 7, 18)

GESTURE_INFO = {
    "fist":    ("FIST",          (30,  80, 255), "Close all fingers"),
    "open":    ("OPEN PALM",     (255, 215, 160), "All 5 fingers extended"),
    "point":   ("POINT",         (50, 245, 255),  "Index finger only"),
    "victory": ("VICTORY",       (70,  225, 70),  "Index + middle fingers"),
    "pinch":   ("PINCH",         (210, 50, 255),  "Thumb + index tip together"),
}

# ── Sliding-window stabilizer ─────────────────────────────────────────────────

class GestureStabilizer:
    """
    Confirms a gesture only when it appears in at least THRESHOLD
    of the last WINDOW raw detections. Prevents flickering on borderline frames.
    """
    WINDOW    = 9   # how many frames to look back
    THRESHOLD = 6   # how many must agree

    def __init__(self):
        self._history   = deque(maxlen=self.WINDOW)
        self.confirmed  = None   # last stable gesture
        self._stable_t  = None   # when confirmed gesture first locked in

    def update(self, raw):
        """raw is the detected gesture string or None."""
        self._history.append(raw)
        if len(self._history) < self.WINDOW:
            return self.confirmed

        counts = {}
        for g in self._history:
            if g is not None:
                counts[g] = counts.get(g, 0) + 1

        best = max(counts, key=counts.get) if counts else None
        if best and counts[best] >= self.THRESHOLD:
            if best != self.confirmed:
                self.confirmed = best
                self._stable_t = time.time()
        elif not counts or max(counts.values()) < self.THRESHOLD:
            if self.confirmed is not None:
                self.confirmed = None
                self._stable_t = None

        return self.confirmed

    def stable_for(self):
        """Seconds since current gesture was first confirmed (0 if none)."""
        if self.confirmed and self._stable_t:
            return time.time() - self._stable_t
        return 0.0

# ── Gesture detection (identical to game) ────────────────────────────────────

def detect_gesture(lm, handedness):
    t_up = lm[4].x < lm[3].x - MARGIN if handedness == "Right" else lm[4].x > lm[3].x + MARGIN
    fingers = [t_up] + [lm[TIPS[i]].y < lm[PIP[i]].y - MARGIN for i in range(1, 5)]
    n = sum(fingers)
    pinch = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) < 0.05
    if pinch:                                                              return "pinch"
    if n == 0:                                                             return "fist"
    if n == 5:                                                             return "open"
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:   return "victory"
    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]: return "point"
    return None

# ── Drawing helpers ───────────────────────────────────────────────────────────

def put(img, txt, pos, scale, color, thick=2):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_landmarks(frame, lm, w, h, color):
    """Draw hand skeleton on frame."""
    connections = [
        (0,1),(1,2),(2,3),(3,4),        # thumb
        (0,5),(5,6),(6,7),(7,8),        # index
        (0,9),(9,10),(10,11),(11,12),   # middle
        (0,13),(13,14),(14,15),(15,16), # ring
        (0,17),(17,18),(18,19),(19,20), # pinky
        (5,9),(9,13),(13,17),           # palm
    ]
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        r = 6 if i in TIPS else 4
        cv2.circle(frame, (px, py), r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), r, (0, 0, 0), 1, cv2.LINE_AA)

def draw_gesture_panel(frame, stable, raw, stabilizer):
    """Right-side info panel."""
    px = W - 340
    cv2.rectangle(frame, (px, 0), (W, H), (12, 8, 20), -1)
    cv2.line(frame, (px, 0), (px, H), (45, 32, 65), 2)

    put(frame, "GESTURE", (px + 20, 48), 0.7, (100, 85, 130), 1)
    put(frame, "VIEWER",  (px + 20, 76), 0.7, (100, 85, 130), 1)
    cv2.line(frame, (px + 10, 90), (W - 10, 90), (45, 32, 65), 1)

    # Stable gesture (large)
    if stable:
        name, color, desc = GESTURE_INFO[stable]
        put(frame, name,  (px + 20, 160), 0.95, color, 2)
        put(frame, desc,  (px + 20, 188), 0.42, (130, 115, 165), 1)
        # Hold duration bar
        held = min(1.0, stabilizer.stable_for() / 3.0)
        bw = W - px - 40
        cv2.rectangle(frame, (px+20, 202), (px+20+bw, 214), (28, 20, 42), -1)
        cv2.rectangle(frame, (px+20, 202), (px+20+int(bw*held), 214), color, -1)
        put(frame, f"{stabilizer.stable_for():.1f}s", (px+20, 228), 0.42, color, 1)
    else:
        put(frame, "---", (px + 20, 160), 0.95, (55, 42, 72), 2)

    cv2.line(frame, (px + 10, 245), (W - 10, 245), (45, 32, 65), 1)

    # Raw detection (smaller, shows flicker)
    put(frame, "raw:", (px + 20, 272), 0.4, (70, 58, 95), 1)
    raw_col = (90, 75, 120) if raw is None else (150, 130, 200)
    put(frame, raw or "none", (px + 65, 272), 0.4, raw_col, 1)

    # All gestures legend
    put(frame, "GESTURES", (px + 20, 320), 0.45, (80, 65, 110), 1)
    cv2.line(frame, (px + 10, 330), (W - 10, 330), (35, 25, 52), 1)
    for i, (key, (name, color, desc)) in enumerate(GESTURE_INFO.items()):
        y = 355 + i * 56
        active = (key == stable)
        bg_col = (30, 20, 48) if active else (16, 10, 26)
        cv2.rectangle(frame, (px+10, y-18), (W-10, y+30), bg_col, -1)
        if active:
            cv2.rectangle(frame, (px+10, y-18), (W-10, y+30), color, 1)
        put(frame, name, (px + 20, y),    0.48, color if active else (80, 65, 105), 1)
        put(frame, desc, (px + 20, y+18), 0.35, (90, 78, 115) if active else (55, 44, 72), 1)

    put(frame, "Q = quit", (px + 20, H - 20), 0.4, (55, 44, 72), 1)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    start_ms   = int(time.time() * 1000)
    stabilizer = GestureStabilizer()

    print("Gesture Viewer  |  Q = quit")

    while True:
        ret, cam = cap.read()
        if not ret:
            break
        cam   = cv2.flip(cam, 1)
        now   = time.time()
        ts_ms = int(now * 1000) - start_ms

        rgb    = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        raw    = None
        lm_data = None
        if result.hand_landmarks:
            lm      = result.hand_landmarks[0]
            hand    = result.handedness[0][0].category_name
            raw     = detect_gesture(lm, hand)
            lm_data = lm

        stable = stabilizer.update(raw)

        # ── Canvas: camera on left, panel on right ────────────────────────
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        cam_w  = W - 340
        cam_resized = cv2.resize(cam, (cam_w, H))
        canvas[:, :cam_w] = cam_resized

        # Landmark skeleton
        if lm_data:
            skel_color = GESTURE_INFO[stable][1] if stable else (120, 100, 160)
            draw_landmarks(canvas, lm_data, cam_w, H, skel_color)

        # Gesture label directly on camera feed
        if stable:
            name, color, _ = GESTURE_INFO[stable]
            put(canvas, name, (30, 60), 1.4, (0, 0, 0), 6)
            put(canvas, name, (30, 60), 1.4, color, 3)

        draw_gesture_panel(canvas, stable, raw, stabilizer)

        cv2.imshow("Gesture Viewer", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
