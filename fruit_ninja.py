"""
Fruit Ninja — hand tracking edition.
Move your open hand to slice fruits; avoid bombs.
Press Q to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math
import os
from collections import deque

# ── MediaPipe ────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    import subprocess
    print("Downloading model...")
    subprocess.run(["curl", "-L", MODEL_URL, "-o", MODEL_PATH], check=True)

_lm_opts = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(_lm_opts)

TIPS = [4, 8, 12, 16, 20]
PIP  = [3, 6, 10, 14, 18]

# ── Constants ─────────────────────────────────────────────────────────────────

W, H           = 1280, 720
GRAVITY        = 0.38
FRUIT_R        = 42
BOMB_R         = 36
TRAIL_LEN      = 22
SLICE_VEL      = 14       # px/frame to trigger slice
SPAWN_BASE     = 1.3      # seconds between spawns (decreases with score)
MAX_LIVES      = 3
COMBO_TIMEOUT  = 1.8      # seconds before combo resets

FRUIT_TYPES = [
    dict(color=(35, 160, 35),  inner=(50, 50, 180),   shine=(120,200,120), label="W", pts=1),
    dict(color=(25, 120, 255), inner=(40, 190, 255),  shine=(130,190,255), label="O", pts=1),
    dict(color=(25, 25, 210),  inner=(60,  80, 255),  shine=(120,120,255), label="A", pts=1),
    dict(color=(20, 210, 220), inner=(50, 255, 255),  shine=(160,240,240), label="L", pts=2),
    dict(color=(150, 30, 130), inner=(200, 70, 180),  shine=(200,130,200), label="G", pts=2),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def seg_hits_circle(p1, p2, cx, cy, r):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    fx, fy = p1[0]-cx,    p1[1]-cy
    a = dx*dx + dy*dy
    if a == 0:
        return math.hypot(fx, fy) < r
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    d = b*b - 4*a*c
    if d < 0:
        return False
    sd = math.sqrt(d)
    return (0 <= (-b-sd)/(2*a) <= 1) or (0 <= (-b+sd)/(2*a) <= 1)


def alpha_rect(img, x1, y1, x2, y2, color, a):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, a, img, 1-a, 0, img)


def put(img, txt, pos, scale, color, thick=2):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)

# ── Game objects ──────────────────────────────────────────────────────────────

class Fruit:
    def __init__(self):
        self.x  = float(random.randint(FRUIT_R+30, W-FRUIT_R-30))
        self.y  = float(H + FRUIT_R + 10)
        self.vx = random.uniform(-4, 4)
        self.vy = random.uniform(-21, -14)
        self.rot   = random.uniform(0, 360)
        self.rot_v = random.uniform(-7, 7)
        self.ft    = random.choice(FRUIT_TYPES)
        self.r     = FRUIT_R
        self.alive  = True
        self.sliced = False
        self.st     = 0.0          # slice elapsed time
        self.side   = random.choice([-1, 1])
        self.juice   = []          # juice particle list

    @property
    def is_bomb(self): return False

    def update(self, dt):
        self.vy += GRAVITY
        self.x  += self.vx
        self.y  += self.vy
        self.rot += self.rot_v
        for p in self.juice:
            p[0] += p[2]; p[1] += p[3]; p[3] += GRAVITY*0.5; p[4] -= dt
        self.juice = [p for p in self.juice if p[4] > 0]
        if self.sliced:
            self.st += dt
            if self.st > 0.55:
                self.alive = False
        elif self.y > H + self.r + 40:
            self.alive = False

    def slice(self):
        self.sliced = True
        self.st = 0.0
        for _ in range(14):
            angle = random.uniform(0, 2*math.pi)
            spd   = random.uniform(2, 8)
            self.juice.append([
                self.x, self.y,
                math.cos(angle)*spd, math.sin(angle)*spd - 3,
                random.uniform(0.3, 0.7)
            ])

    def draw(self, frame):
        ix, iy = int(self.x), int(self.y)
        # juice particles
        for p in self.juice:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, self.ft["inner"], -1, cv2.LINE_AA)

        if self.sliced:
            fade = max(0.0, 1.0 - self.st * 2)
            col  = tuple(int(c * fade) for c in self.ft["inner"])
            off  = int(self.side * self.st * 55)
            cv2.ellipse(frame, (ix - off, iy + int(self.st*25)),
                        (self.r, self.r), self.rot, 0, 180, col, -1, cv2.LINE_AA)
            cv2.ellipse(frame, (ix + off, iy - int(self.st*25)),
                        (self.r, self.r), self.rot, 180, 360, col, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (ix, iy), self.r, self.ft["color"], -1, cv2.LINE_AA)
            cv2.circle(frame, (ix - self.r//4, iy - self.r//4),
                       self.r//3, self.ft["shine"], -1, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), self.r, (0, 0, 0), 2, cv2.LINE_AA)


class Bomb:
    def __init__(self):
        self.x  = float(random.randint(BOMB_R+30, W-BOMB_R-30))
        self.y  = float(H + BOMB_R + 10)
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-19, -13)
        self.rot   = 0.0
        self.rot_v = random.uniform(-9, 9)
        self.r     = BOMB_R
        self.alive  = True
        self.sliced = False
        self.st     = 0.0

    @property
    def is_bomb(self): return True

    def update(self, dt):
        self.vy += GRAVITY
        self.x  += self.vx
        self.y  += self.vy
        self.rot += self.rot_v
        if self.sliced:
            self.st += dt
            if self.st > 0.5:
                self.alive = False
        elif self.y > H + self.r + 40:
            self.alive = False

    def slice(self):
        self.sliced = True
        self.st = 0.0

    def draw(self, frame):
        ix, iy = int(self.x), int(self.y)
        if self.sliced:
            r_exp = int(self.r + self.st * 120)
            fade  = max(0, 1.0 - self.st * 2.5)
            col   = (int(30*fade), int(80*fade), int(255*fade))
            cv2.circle(frame, (ix, iy), r_exp, col, -1, cv2.LINE_AA)
            put(frame, "BOOM!", (ix-38, iy+8), 0.9, (255,255,255), 2)
        else:
            cv2.circle(frame, (ix, iy), self.r, (18, 18, 18), -1, cv2.LINE_AA)
            fuse_angle = math.radians(self.rot - 90)
            fx = ix + int(math.cos(fuse_angle) * (self.r + 14))
            fy = iy + int(math.sin(fuse_angle) * (self.r + 14))
            cv2.line(frame, (ix, iy - self.r), (fx, fy), (60,150,255), 3, cv2.LINE_AA)
            cv2.circle(frame, (fx, fy), 5, (30, 210, 255), -1, cv2.LINE_AA)
            put(frame, "BOMB", (ix-26, iy+6), 0.46, (70, 70, 70), 2)
            cv2.circle(frame, (ix, iy), self.r, (50, 50, 50), 2, cv2.LINE_AA)

# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_heart(frame, cx, cy, sz, filled):
    col = (60, 60, 220) if filled else (50, 50, 60)
    pts = np.array([[
        [cx + int(16*math.sin(math.radians(i))**3 * sz/16),
         cy + int(-(13*math.cos(math.radians(i))
                    - 5*math.cos(math.radians(2*i))
                    - 2*math.cos(math.radians(3*i))
                    - math.cos(math.radians(4*i))) * sz/16)]
        for i in range(360)]], dtype=np.int32)
    if filled:
        cv2.fillPoly(frame, pts, col, cv2.LINE_AA)
    cv2.polylines(frame, pts, True, (20,20,120) if filled else (40,40,50), 1, cv2.LINE_AA)


def draw_hud(frame, score, lives, combo, highscore):
    alpha_rect(frame, 0, 0, W, 58, (8, 8, 12), 0.72)
    put(frame, f"SCORE  {score:05d}", (18, 40), 1.0, (220, 200, 255), 2)
    put(frame, f"BEST {highscore:05d}", (310, 40), 0.65, (120, 100, 160), 1)
    if combo > 1:
        put(frame, f"x{combo}  COMBO!", (W//2 - 110, 40), 1.1, (60, 230, 255), 3)
    for i in range(MAX_LIVES):
        draw_heart(frame, W - 42 - i*46, 28, 15, i < lives)


def draw_trail(frame, trail, vel):
    n = len(trail)
    if n < 2:
        return
    slicing = vel > SLICE_VEL
    for i in range(1, n):
        a = i / n
        if slicing:
            color = (int(40*a), int(190*a), int(255*a))
            thick = max(1, int(7*a))
        else:
            color = (int(55*a), int(55*a), int(65*a))
            thick = max(1, int(3*a))
        cv2.line(frame, trail[i-1], trail[i], color, thick, cv2.LINE_AA)
    if slicing:
        cv2.circle(frame, trail[-1], 9, (180, 240, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, trail[-1], 9, (255, 255, 255), 1,  cv2.LINE_AA)


def draw_popups(frame, popups, now):
    alive = []
    for px, py, pts, t in popups:
        age = now - t
        if age < 1.0:
            y_off = int(age * 70)
            fade  = int((1.0 - age) * 255)
            col   = (0, fade, int(fade * 0.4))
            put(frame, f"+{pts}", (px, py - y_off), 1.3, col, 3)
            alive.append((px, py, pts, t))
    return alive


def draw_countdown(frame, n):
    alpha_rect(frame, 0, 0, W, H, (0, 0, 0), 0.55)
    put(frame, str(n), (W//2 - 40, H//2 + 50), 5.0, (220, 200, 255), 8)
    put(frame, "Get ready!", (W//2 - 140, H//2 - 80), 1.4, (180, 180, 180), 2)


def draw_game_over(frame, score, highscore):
    alpha_rect(frame, 0, 0, W, H, (5, 0, 10), 0.65)
    put(frame, "GAME  OVER", (W//2 - 240, H//2 - 70), 2.6, (80, 60, 230), 5)
    put(frame, f"Score:  {score:05d}", (W//2 - 130, H//2 + 20), 1.3, (220,200,255), 2)
    put(frame, f"Best:   {highscore:05d}", (W//2 - 130, H//2 + 68), 1.3, (100,220,255), 2)
    put(frame, "Open hand to play again", (W//2 - 215, H//2 + 140), 0.9, (160,160,160), 1)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    start_ms  = int(time.time() * 1000)
    prev_time = time.time()

    # state
    objects      = []
    trail        = deque(maxlen=TRAIL_LEN)
    prev_palm    = None
    velocity     = 0.0
    score        = 0
    highscore    = 0
    lives        = MAX_LIVES
    combo        = 0
    combo_end    = 0.0
    next_spawn   = 0.0
    popups       = []
    game_over    = False
    open_frames  = 0       # for restart gesture
    countdown    = 3       # 3-2-1 before game starts
    cd_end       = time.time() + 1.0

    print("Fruit Ninja  |  move your hand to slice  |  Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now   = time.time()
        dt    = max(now - prev_time, 1e-4)
        prev_time = now
        ts_ms = int(now * 1000) - start_ms

        # ── Hand detection ──────────────────────────────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect_for_video(mp_img, ts_ms)

        palm_pos    = None
        fingers_up  = 0
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            px = int(sum(lm[i].x for i in [0,5,9,13,17]) / 5 * W)
            py = int(sum(lm[i].y for i in [0,5,9,13,17]) / 5 * H)
            palm_pos   = (px, py)
            fingers_up = sum(1 for i in range(1,5) if lm[TIPS[i]].y < lm[PIP[i]].y - 0.02)
            trail.append(palm_pos)
            if prev_palm:
                dx = px - prev_palm[0]
                dy = py - prev_palm[1]
                velocity = math.hypot(dx, dy)
            prev_palm = palm_pos
        else:
            prev_palm = None
            velocity  = 0.0

        # ── Game canvas (dark, not camera) ─────────────────────────────────
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (12, 8, 18)   # deep dark purple-black

        # ── Camera PiP in bottom-right corner ──────────────────────────────
        pip_w, pip_h = 240, 135
        pip = cv2.resize(frame, (pip_w, pip_h))
        # dim the pip so it doesn't distract
        pip = (pip * 0.55).astype(np.uint8)
        px0, py0 = W - pip_w - 12, H - pip_h - 12
        canvas[py0:py0+pip_h, px0:px0+pip_w] = pip
        cv2.rectangle(canvas, (px0-1, py0-1), (px0+pip_w, py0+pip_h), (60,50,80), 1)

        # ── Countdown ──────────────────────────────────────────────────────
        if countdown > 0:
            if now >= cd_end:
                countdown -= 1
                cd_end = now + 1.0
                if countdown == 0:
                    next_spawn = now + 0.3
            for obj in objects:
                obj.draw(canvas)
            draw_trail(canvas, list(trail), velocity)
            draw_hud(canvas, score, lives, combo, highscore)
            if countdown > 0:
                draw_countdown(canvas, countdown)
            cv2.imshow("Fruit Ninja", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ── Game over ──────────────────────────────────────────────────────
        if game_over:
            for obj in objects:
                obj.draw(canvas)
            draw_trail(canvas, list(trail), velocity)
            draw_hud(canvas, score, lives, combo, highscore)
            draw_game_over(canvas, score, highscore)
            # restart: hold open hand for 20 frames
            if fingers_up >= 4:
                open_frames += 1
            else:
                open_frames = 0
            if open_frames >= 20:
                objects, score, lives = [], 0, MAX_LIVES
                combo, combo_end = 0, 0.0
                popups = []; trail.clear()
                next_spawn  = now + 0.5
                game_over   = False
                open_frames = 0
                countdown   = 3
                cd_end      = now + 1.0
            cv2.imshow("Fruit Ninja", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ── Spawn ───────────────────────────────────────────────────────────
        if now >= next_spawn:
            is_bomb = random.random() < 0.14
            objects.append(Bomb() if is_bomb else Fruit())
            # speed up as score grows
            interval = max(0.55, SPAWN_BASE - score * 0.003)
            next_spawn = now + random.uniform(interval * 0.7, interval * 1.35)

        # ── Slice detection ─────────────────────────────────────────────────
        if velocity > SLICE_VEL and len(trail) >= 2:
            p1, p2 = list(trail)[-2], list(trail)[-1]
            for obj in objects:
                if not obj.alive or obj.sliced:
                    continue
                if seg_hits_circle(p1, p2, obj.x, obj.y, obj.r):
                    obj.slice()
                    if obj.is_bomb:
                        lives -= 1
                        combo  = 0
                    else:
                        combo += 1
                        combo_end = now + COMBO_TIMEOUT
                        bonus = obj.ft["pts"] * (combo if combo > 1 else 1)
                        score += bonus
                        popups.append((int(obj.x), int(obj.y), bonus, now))

        # ── Combo timeout ───────────────────────────────────────────────────
        if now > combo_end:
            combo = 0

        # ── Update objects ──────────────────────────────────────────────────
        for obj in objects:
            obj.update(dt)
            # fruit fell off → lose life
            if not obj.alive and not obj.sliced and not obj.is_bomb:
                lives -= 1
                combo  = 0

        objects = [o for o in objects if o.alive]

        if lives <= 0:
            lives     = 0
            game_over = True
            highscore = max(highscore, score)

        # ── Draw everything ─────────────────────────────────────────────────
        for obj in objects:
            obj.draw(canvas)

        draw_trail(canvas, list(trail), velocity)
        draw_hud(canvas, score, lives, combo, highscore)
        popups = draw_popups(canvas, popups, now)

        cv2.imshow("Fruit Ninja", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
