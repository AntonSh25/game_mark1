"""
Spell Caster — defend your castle with hand gestures.
  fist      → Fireball    (AoE blast at cursor)
  1 finger  → Lightning   (instant beam across cursor height)
  open hand → Frost Nova  (AoE freeze at cursor)
  victory   → Wind Slash  (knockback wave)
  pinch     → Vortex      (2 s pull, setup for combos)

Combos (cast two spells within 2.5 s):
  Vortex  → Fireball   = INFERNO  (giant explosion)
  Frost   → Lightning  = SHATTER  (instant-kill frozen)
  Fire    → Fire       = METEOR   (huge AoE, 8 dmg)
  Wind    → Vortex     = TORNADO  (mega knockback)

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

# ── MediaPipe ─────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    import subprocess
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
TIPS = [4, 8, 12, 16, 20]
PIP  = [3, 6, 10, 14, 18]
MARGIN = 0.02

# ── Constants ─────────────────────────────────────────────────────────────────

W, H           = 1280, 720
CASTLE_X       = 95
CASTLE_HP_MAX  = 15
CHARGE_FRAMES  = 20

C_FIRE   = (30,  80, 255)
C_LIGHT  = (50, 245, 255)
C_FROST  = (255, 215, 160)
C_WIND   = (70,  225,  70)
C_VORTEX = (210,  50, 255)
C_DARK   = (10,   7,  18)

SPELLS = {
    "fist":    dict(name="Fireball",   color=C_FIRE,   dmg=3, kind="blast",   hint="[fist]"),
    "point":   dict(name="Lightning",  color=C_LIGHT,  dmg=2, kind="beam",    hint="[1 finger]"),
    "open":    dict(name="Frost Nova", color=C_FROST,  dmg=1, kind="nova",    hint="[palm]"),
    "victory": dict(name="Wind Slash", color=C_WIND,   dmg=1, kind="wave",    hint="[V]"),
    "pinch":   dict(name="Vortex",     color=C_VORTEX, dmg=0, kind="vortex",  hint="[pinch]"),
}

# (prev_spell, cur_spell) → (display_name, color, kind_override)
COMBOS = {
    ("pinch",   "fist"):    ("INFERNO",  C_FIRE,   "inferno"),
    ("open",    "point"):   ("SHATTER",  C_FROST,  "shatter"),
    ("fist",    "fist"):    ("METEOR",   (0, 40, 190), "meteor"),
    ("victory", "pinch"):   ("TORNADO",  C_WIND,   "tornado"),
}

ENEMY_DEFS = {
    #          hp  spd   color           w   h   pts  immune            weak
    "goblin":  (2,  1.0, (35, 130, 35),  26, 42, 10,  set(),           set()),
    "orc":     (5,  0.6, (35,  70, 160), 36, 54, 25,  set(),           set()),
    "golem":   (7,  0.4, (25,  65, 190), 44, 62, 40,  {"fireball"},    {"frost", "lightning"}),
    "wraith":  (4,  1.1, (170, 170, 25), 28, 52, 30,  {"frost"},       {"fireball"}),
    "shaman":  (3,  0.5, (155, 35, 155), 30, 48, 35,  set(),           set()),
}

WAVE_DEFS = [
    [("goblin", 6)],
    [("goblin", 4), ("orc", 2)],
    [("orc", 3), ("wraith", 2)],
    [("goblin", 3), ("golem", 2)],
    [("orc", 2), ("wraith", 3), ("shaman", 1)],
    [("goblin", 3), ("golem", 2), ("shaman", 2)],
    [("orc", 3), ("golem", 2), ("wraith", 3)],
    [("orc", 4), ("golem", 3), ("wraith", 2), ("shaman", 2)],
]

def wave_config(n):
    base = WAVE_DEFS[n % len(WAVE_DEFS)]
    mult = 1 + n // len(WAVE_DEFS)
    return [(t, c * mult) for t, c in base]

# ── Enemy ─────────────────────────────────────────────────────────────────────

class Enemy:
    def __init__(self, etype, y=None):
        hp, spd, color, w, h, pts, immune, weak = ENEMY_DEFS[etype]
        self.type   = etype
        self.x      = float(W + w + 10)
        self.y      = float(y or random.randint(110, H - 110))
        self.hp     = hp;  self.max_hp = hp
        self.spd    = spd; self.color  = color
        self.w      = w;   self.h      = h
        self.pts    = pts
        self.immune = immune;  self.weak = weak
        self.heals  = (etype == "shaman")
        self.alive  = True
        self.scored = False
        self.frozen = 0.0
        self.pull_target = None
        self.flash  = 0.0

    # Returns damage actually dealt (0 = immune)
    def take_damage(self, dmg, dtype):
        if dtype in self.immune:
            return 0
        if dtype in self.weak:
            dmg *= 2
        self.hp -= dmg
        self.flash = 0.15
        if self.hp <= 0:
            self.alive = False
        return dmg

    def update(self, dt, all_enemies):
        self.flash  = max(0.0, self.flash - dt)
        if self.frozen > 0:
            self.frozen -= dt
            return
        if self.pull_target:
            tx, ty = self.pull_target
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist > 6:
                s = self.spd * 3.5
                self.x += dx / dist * s
                self.y += dy / dist * s
        else:
            self.x -= self.spd

        if self.heals:
            near = min(
                (e for e in all_enemies if e is not self and e.alive),
                key=lambda e: math.hypot(e.x - self.x, e.y - self.y),
                default=None,
            )
            if near and near.hp < near.max_hp:
                near.hp = min(near.max_hp, near.hp + 0.25 * dt)

    def draw(self, frame):
        ix, iy = int(self.x), int(self.y)
        hw, hh = self.w // 2, self.h // 2
        head_r = max(hh // 3, 7)

        col = list(self.color)
        if self.flash > 0:
            col = [min(255, c + 160) for c in col]
        if self.frozen > 0:
            col = [min(255, col[0] + 30), min(255, col[1] + 30), min(255, col[2] + 120)]
        col = tuple(col)

        # Body
        cv2.rectangle(frame, (ix - hw, iy - hh + head_r * 2),
                      (ix + hw, iy + hh), col, -1, cv2.LINE_AA)
        # Head
        cv2.circle(frame, (ix, iy - hh + head_r), head_r, col, -1, cv2.LINE_AA)
        # Eyes
        ex = head_r // 2
        cv2.circle(frame, (ix - ex, iy - hh + head_r - 2), 2, (0, 0, 0), -1)
        cv2.circle(frame, (ix + ex, iy - hh + head_r - 2), 2, (0, 0, 0), -1)
        # Outline
        cv2.rectangle(frame, (ix - hw, iy - hh + head_r * 2),
                      (ix + hw, iy + hh), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (ix, iy - hh + head_r), head_r, (0, 0, 0), 1, cv2.LINE_AA)

        # HP bar
        bw = self.w + 12
        bx, by = ix - bw // 2, iy - hh - 8
        cv2.rectangle(frame, (bx, by - 5), (bx + bw, by), (40, 15, 15), -1)
        fill = int(bw * max(0, self.hp) / self.max_hp)
        cv2.rectangle(frame, (bx, by - 5), (bx + fill, by), (40, 180, 40), -1)

        # Status icons
        if self.frozen > 0:
            cv2.putText(frame, "~", (ix - 4, iy - hh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 255), 1, cv2.LINE_AA)
        if self.heals:
            cv2.circle(frame, (ix + hw + 5, iy - hh + 5), 5, (40, 200, 80), -1)
        if self.immune:
            cv2.circle(frame, (ix - hw - 5, iy - hh + 5), 4, (60, 60, 200), -1)

# ── Particles & effects ───────────────────────────────────────────────────────

class Particle:
    __slots__ = ("x","y","vx","vy","color","size","life","max_life","alive")
    def __init__(self, x, y, vx, vy, color, size, life):
        self.x=x; self.y=y; self.vx=vx; self.vy=vy
        self.color=color; self.size=size
        self.life=life; self.max_life=life; self.alive=True
    def update(self, dt):
        self.x+=self.vx; self.y+=self.vy; self.vy+=0.25
        self.life-=dt
        if self.life<=0: self.alive=False
    def draw(self, f):
        a=self.life/self.max_life
        col=tuple(int(c*a) for c in self.color)
        cv2.circle(f,(int(self.x),int(self.y)),max(1,int(self.size*a)),col,-1,cv2.LINE_AA)

class Effect:
    def __init__(self, kind, x, y, color, dur, **kw):
        self.kind=kind; self.x=x; self.y=y; self.color=color
        self.dur=dur; self.t=0.0; self.alive=True
        self.__dict__.update(kw)
    def update(self, dt):
        self.t+=dt
        if self.t>=self.dur: self.alive=False
    def draw(self, f):
        a=max(0.0, 1.0-self.t/self.dur)
        col=tuple(int(c*a) for c in self.color)
        k=self.kind
        if k=="blast" or k=="inferno" or k=="meteor":
            r=int(getattr(self,"radius",80)*min(1.0, self.t/(self.dur*0.35)))
            if self.t>self.dur*0.35:
                r=int(getattr(self,"radius",80)*(1.0-(self.t-self.dur*0.35)/(self.dur*0.65)))
            cv2.circle(f,(int(self.x),int(self.y)),max(1,r),col,-1,cv2.LINE_AA)
            cv2.circle(f,(int(self.x),int(self.y)),max(1,r+5),(255,255,255),2,cv2.LINE_AA)
        elif k=="beam" or k=="shatter":
            th=max(1,int(7*a))
            cv2.line(f,(int(self.x),int(self.y)),(W,int(self.y)),col,th,cv2.LINE_AA)
            cv2.line(f,(int(self.x),int(self.y)),(W,int(self.y)),(255,255,255),max(1,th//3),cv2.LINE_AA)
        elif k=="nova":
            r=int(getattr(self,"radius",150)*self.t/self.dur)
            cv2.circle(f,(int(self.x),int(self.y)),max(1,r),col,3,cv2.LINE_AA)
            cv2.circle(f,(int(self.x),int(self.y)),max(1,r//2),col,1,cv2.LINE_AA)
        elif k=="wave" or k=="tornado":
            wx=int(self.x+self.t/self.dur*(W-self.x))
            width=max(1,int((14 if k=="tornado" else 8)*a))
            cv2.line(f,(wx,55),(wx,H-20),col,width,cv2.LINE_AA)
        elif k=="vortex":
            for ring in range(3):
                r=int(getattr(self,"radius",200)*(0.25+ring*0.38)*(0.8+0.2*math.sin(self.t*7+ring)))
                cv2.circle(f,(int(self.x),int(self.y)),max(1,r),col,max(1,2-ring),cv2.LINE_AA)

# ── Spell logic ───────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "blast":"fireball","beam":"lightning","nova":"frost","wave":"wind",
    "vortex":"vortex","inferno":"fireball","shatter":"frost","meteor":"fireball","tornado":"wind",
}

def burst(particles, x, y, color, n=14, spd=6):
    for _ in range(n):
        a=random.uniform(0,2*math.pi); s=random.uniform(1,spd)
        particles.append(Particle(x,y,math.cos(a)*s,math.sin(a)*s-2,
                                  color,random.randint(4,10),random.uniform(0.35,0.75)))

def cast(spell_key, cx, cy, enemies, particles, effects, combo_kind=None):
    """Fire a spell. Returns score earned."""
    kind  = combo_kind or SPELLS[spell_key]["kind"]
    color = SPELLS[spell_key]["color"]
    dmg   = SPELLS[spell_key]["dmg"]
    dtype = DTYPE_MAP.get(kind, kind)
    earned = 0

    if kind in ("blast", "inferno", "meteor"):
        r   = 200 if kind == "inferno" else (260 if kind == "meteor" else 90)
        dmg = 8   if kind == "meteor"  else (dmg * 3 if kind == "inferno" else dmg)
        for e in enemies:
            if not e.alive: continue
            if math.hypot(e.x-cx, e.y-cy) < r:
                e.take_damage(dmg, dtype)
                burst(particles, e.x, e.y, color, 12, 5)
                if not e.alive and not e.scored:
                    earned += e.pts; e.scored = True
        effects.append(Effect("blast", cx, cy, color, 0.55, radius=r))
        burst(particles, cx, cy, color, 22, 9)

    elif kind in ("beam", "shatter"):
        radius = 38
        for e in enemies:
            if not e.alive: continue
            if abs(e.y - cy) < radius + e.h // 2:
                d = 9999 if (kind == "shatter" and e.frozen > 0) else dmg
                e.take_damage(d, dtype)
                burst(particles, e.x, e.y, color, 8, 4)
                if not e.alive and not e.scored:
                    earned += e.pts; e.scored = True
        effects.append(Effect("beam", cx, cy, color, 0.42))

    elif kind == "nova":
        r = 165
        for e in enemies:
            if not e.alive: continue
            if math.hypot(e.x-cx, e.y-cy) < r:
                e.take_damage(dmg, dtype)
                e.frozen = 2.8
                if not e.alive and not e.scored:
                    earned += e.pts; e.scored = True
        effects.append(Effect("nova", cx, cy, color, 0.65, radius=r))

    elif kind in ("wave", "tornado"):
        radius = 55; push = 150 if kind == "tornado" else 80
        for e in enemies:
            if not e.alive: continue
            if abs(e.y - cy) < radius:
                e.take_damage(dmg, dtype)
                e.x = min(W + e.w, e.x + push)
                burst(particles, e.x, e.y, color, 7, 4)
                if not e.alive and not e.scored:
                    earned += e.pts; e.scored = True
        effects.append(Effect("wave", cx, cy, color, 0.5))

    elif kind == "vortex":
        r = 260
        for e in enemies:
            if not e.alive: continue
            if math.hypot(e.x-cx, e.y-cy) < r:
                e.pull_target = (cx, cy)
        effects.append(Effect("vortex", cx, cy, color, 2.0, radius=r))

    return earned

# ── Gesture detection ──────────────────────────────────────────────────────────

def detect_gesture(lm, handedness):
    t_up = lm[4].x < lm[3].x - MARGIN if handedness == "Right" else lm[4].x > lm[3].x + MARGIN
    fingers = [t_up] + [lm[TIPS[i]].y < lm[PIP[i]].y - MARGIN for i in range(1, 5)]
    n = sum(fingers)
    pinch = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) < 0.05
    if pinch:                                             return "pinch"
    if n == 0:                                            return "fist"
    if n == 5:                                            return "open"
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]: return "victory"
    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]: return "point"
    return None

# ── Drawing helpers ───────────────────────────────────────────────────────────

def arect(img, x1, y1, x2, y2, color, a):
    ov=img.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,a,img,1-a,0,img)

def put(img, txt, pos, scale, color, thick=2):
    cv2.putText(img,txt,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)

def draw_castle(frame, hp, mx):
    cx = CASTLE_X
    stone = (65, 52, 38)
    # Towers
    for tx in (cx-18, cx+18):
        cv2.rectangle(frame, (tx-12,150),(tx+12,H-75), stone, -1)
        for m in range(3):
            cv2.rectangle(frame, (tx-12+m*9,140),(tx-5+m*9,160), stone, -1)
    # Main wall
    cv2.rectangle(frame, (cx-30,220),(cx+30,H-75), stone, -1)
    for m in range(4):
        cv2.rectangle(frame, (cx-28+m*15,210),(cx-16+m*15,228), stone, -1)
    # Door
    cv2.ellipse(frame,(cx,H-75),(16,22),0,180,360,(15,10,8),-1)
    # HP bar inside wall
    bh=int((H-310)*max(0,hp)/mx)
    cv2.rectangle(frame,(cx-8,250),(cx+8,H-100),(35,15,15),-1)
    col=(40,180,40) if hp>mx//3 else (40,80,210)
    cv2.rectangle(frame,(cx-8,H-100-bh),(cx+8,H-100),col,-1)
    put(frame,f"{hp}",(cx-10,244),0.46,(200,190,210),1)

def draw_cursor(frame, x, y, gesture, charge):
    if gesture in SPELLS:
        col = SPELLS[gesture]["color"]
        r   = int(18 + charge * 16)
        cv2.circle(frame,(x,y),r,col,2,cv2.LINE_AA)
        cv2.circle(frame,(x,y),4,col,-1,cv2.LINE_AA)
        if charge > 0:
            ang = int(360*charge)
            cv2.ellipse(frame,(x,y),(r+5,r+5),-90,0,ang,col,3,cv2.LINE_AA)
        # Spell name near cursor
        put(frame, SPELLS[gesture]["name"], (x+r+8, y+6), 0.5, col, 1)
    else:
        cv2.circle(frame,(x,y),12,(70,60,90),1,cv2.LINE_AA)
        cv2.circle(frame,(x,y),3,(110,90,130),-1,cv2.LINE_AA)

def draw_spell_bar(frame, cur_gesture, charge, cooldown):
    y0 = H - 72
    arect(frame, 0, y0-4, W, H, (8,5,14), 0.82)
    for i, (key, sp) in enumerate(SPELLS.items()):
        bx = 10 + i * 118
        active = (key == cur_gesture and cooldown <= 0)
        bg = (50,35,70) if active else (25,18,38)
        cv2.rectangle(frame,(bx,y0),(bx+113,y0+64),bg,-1)
        border_col = sp["color"] if active else (55,40,70)
        cv2.rectangle(frame,(bx,y0),(bx+113,y0+64),border_col,2 if active else 1)
        put(frame, sp["name"],  (bx+6, y0+20), 0.44, sp["color"], 1)
        put(frame, sp["hint"],  (bx+6, y0+37), 0.36, (100,85,125), 1)
        # Charge bar
        if active:
            cw = int(101*charge)
            cv2.rectangle(frame,(bx+6,y0+46),(bx+6+cw,y0+58),sp["color"],-1)
        cv2.rectangle(frame,(bx+6,y0+46),(bx+107,y0+58),(55,40,70),1)
        # Combo hints
    # Combo reference
    cx_ref = 10 + len(SPELLS)*118 + 20
    arect(frame, cx_ref-4, y0, cx_ref+230, H, (15,10,22), 0.5)
    put(frame,"COMBOS", (cx_ref,y0+16),0.38,(120,100,160),1)
    combos_brief=[("Vortex+Fire","INFERNO"),("Frost+Bolt","SHATTER"),
                  ("Fire+Fire","METEOR"),("Wind+Vortex","TORNADO")]
    for j,(a,b) in enumerate(combos_brief):
        put(frame,f"{a} = {b}",(cx_ref, y0+30+j*14),0.32,(90,75,120),1)

def draw_top_hud(frame, score, wave, combo_text, combo_t, now):
    arect(frame,0,0,W,55,(6,4,12),0.80)
    put(frame,f"SCORE  {score:06d}",(140,38),0.9,(200,180,255),2)
    put(frame,f"WAVE  {wave+1}",(440,38),0.9,(170,155,240),2)
    if combo_text and now-combo_t < 2.0:
        a = 1.0-(now-combo_t)/2.0
        col=tuple(int(c*a) for c in (60,240,255))
        put(frame, f"** {combo_text} **", (W//2-160, H//2+10), 2.2, col, 5)

def draw_pip(frame, cam):
    pw,ph=210,118
    pip=(cv2.resize(cam,(pw,ph))*0.48).astype(np.uint8)
    px0,py0=W-pw-10,H-ph-80
    frame[py0:py0+ph,px0:px0+pw]=pip
    cv2.rectangle(frame,(px0-1,py0-1),(px0+pw,py0+ph),(70,50,90),1)

def draw_wave_banner(frame, wave, t, now):
    age = now-t
    if age > 2.8: return
    a = min(1.0, age*3) if age<0.4 else max(0.0,1.0-(age-1.4)/1.4)
    col=tuple(int(c*a) for c in (180,155,255))
    put(frame, f"— WAVE  {wave+1} —",(W//2-165,H//2+12),2.2,col,5)

def draw_between_waves(frame, wave, t_left):
    put(frame,f"Wave {wave+2} in {t_left:.0f}s...",(W//2-175,H//2+10),1.1,(140,120,190),2)

def draw_game_over(frame, score, best):
    arect(frame,0,0,W,H,(4,0,8),0.72)
    put(frame,"CASTLE  FALLEN",(W//2-295,H//2-65),2.5,(55,50,220),6)
    put(frame,f"Score:  {score:06d}",(W//2-130,H//2+20),1.3,(200,180,255),2)
    put(frame,f"Best:   {best:06d}", (W//2-130,H//2+65),1.3,(70,215,255),2)
    put(frame,"Open hand to restart",(W//2-195,H//2+130),0.9,(140,130,175),1)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    start_ms  = int(time.time()*1000)
    prev_time = time.time()

    enemies, particles, effects = [], [], []
    score = 0;  highscore = 0;  castle_hp = CASTLE_HP_MAX
    wave  = 0;  wave_queue = []
    between_waves = True;  wave_start_t = time.time()
    wave_banner_t = time.time()-20;  spawn_timer = 0.0

    palm_x, palm_y = W//2, H//2
    cur_gesture    = None
    charge_frames  = 0
    cooldown_until = 0.0
    last_spell     = None;  last_spell_t = 0.0
    combo_text     = "";    combo_t      = 0.0
    vortex_until   = 0.0

    game_over    = False;  open_frames = 0
    countdown    = 3;      cd_end = time.time()+1.0

    print("Spell Caster  |  defend your castle  |  Q = quit")

    while True:
        ret, cam = cap.read()
        if not ret: break
        cam = cv2.flip(cam, 1)
        now = time.time()
        dt  = max(now-prev_time, 1e-4)
        prev_time = now
        ts_ms = int(now*1000)-start_ms

        # ── Hand tracking ────────────────────────────────��───────────────────
        rgb    = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        raw_gesture = None;  fingers_open = 0;  hand_found = False
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            hand_found = True
            palm_x = int(sum(lm[i].x for i in [0,5,9,13,17])/5*W)
            palm_y = int(sum(lm[i].y for i in [0,5,9,13,17])/5*H)
            hand   = result.handedness[0][0].category_name
            raw_gesture  = detect_gesture(lm, hand)
            fingers_open = sum(1 for i in range(1,5) if lm[TIPS[i]].y < lm[PIP[i]].y-MARGIN)

        # ── Canvas ───────────────────────────────────────────────────────────
        canvas = np.full((H, W, 3), C_DARK, dtype=np.uint8)

        # ── Countdown ────────────────────────────────────────────────────────
        if countdown > 0:
            if now >= cd_end:
                countdown -= 1; cd_end = now+1.0
            draw_castle(canvas, castle_hp, CASTLE_HP_MAX)
            draw_top_hud(canvas, score, wave, combo_text, combo_t, now)
            draw_pip(canvas, cam)
            if countdown > 0:
                arect(canvas,0,0,W,H,(0,0,0),0.55)
                put(canvas,str(countdown),(W//2-35,H//2+50),5.0,(200,180,255),8)
                put(canvas,"Prepare your spells!",(W//2-200,H//2-80),1.2,(160,140,200),2)
            cv2.imshow("Spell Caster", canvas)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue

        # ── Game over ─────────────────────────────────────────────────────────
        if game_over:
            for obj in (*enemies, *particles, *effects): obj.draw(canvas)
            draw_castle(canvas, 0, CASTLE_HP_MAX)
            draw_top_hud(canvas, score, wave, combo_text, combo_t, now)
            draw_game_over(canvas, score, highscore)
            draw_pip(canvas, cam)
            open_frames = open_frames+1 if fingers_open>=4 else 0
            if open_frames >= 20:
                enemies,particles,effects=[],[],[]
                score=0; castle_hp=CASTLE_HP_MAX; wave=0; wave_queue=[]
                between_waves=True; wave_start_t=now; wave_banner_t=now-20
                cur_gesture=None; charge_frames=0; cooldown_until=0.0
                last_spell=None; last_spell_t=0.0; combo_text=""; combo_t=0.0
                vortex_until=0.0; game_over=False; open_frames=0
                countdown=3; cd_end=now+1.0
            cv2.imshow("Spell Caster", canvas)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue

        # ── Wave management ───────────────────────────────────────────────────
        if between_waves and not enemies:
            if now-wave_start_t > 3.0 or wave == 0:
                cfg = wave_config(wave)
                wave_queue = [t for (t,c) in cfg for _ in range(c)]
                random.shuffle(wave_queue)
                between_waves = False
                spawn_timer   = now+0.6
                wave_banner_t = now

        if not between_waves and wave_queue and now >= spawn_timer:
            enemies.append(Enemy(wave_queue.pop(0), random.randint(110,H-110)))
            spawn_timer = now+random.uniform(0.7,1.5)

        # ── Vortex release ────────────────────────────────────────────────────
        if vortex_until > 0 and now > vortex_until:
            for e in enemies: e.pull_target = None
            vortex_until = 0.0

        # ── Gesture → charge → cast ───────────────────────────────────────────
        can_cast = hand_found and now >= cooldown_until
        if can_cast and raw_gesture in SPELLS:
            if raw_gesture == cur_gesture:
                charge_frames += 1
            else:
                cur_gesture   = raw_gesture
                charge_frames = 1

            if charge_frames >= CHARGE_FRAMES:
                # Check combo
                combo_key = (last_spell, cur_gesture)
                if last_spell and (now-last_spell_t < 2.5) and combo_key in COMBOS:
                    cname, ccol, ckind = COMBOS[combo_key]
                    score      += cast(cur_gesture, palm_x, palm_y, enemies, particles, effects,
                                       combo_kind=ckind) * 3 + 60
                    combo_text  = cname
                    combo_t     = now
                    last_spell  = None
                else:
                    pts = cast(cur_gesture, palm_x, palm_y, enemies, particles, effects)
                    score      += pts * 10
                    if cur_gesture == "pinch":
                        vortex_until = now+2.0
                    last_spell   = cur_gesture
                    last_spell_t = now

                cooldown_until = now+0.45
                charge_frames  = 0
                cur_gesture    = None
        else:
            if raw_gesture != cur_gesture:
                cur_gesture   = raw_gesture
                charge_frames = 0

        charge = min(1.0, charge_frames/CHARGE_FRAMES) if can_cast else 0.0

        # ── Update ────────────────────────────────────────────────────────────
        for e in enemies:
            e.update(dt, enemies)
            if e.alive and e.x < CASTLE_X+28:
                castle_hp -= 1
                e.alive = False
                burst(particles, CASTLE_X+30, e.y, (40,70,220), 10, 5)

        for p  in particles: p.update(dt)
        for ef in effects:   ef.update(dt)

        enemies   = [e  for e  in enemies   if e.alive  or e.flash > 0]
        particles = [p  for p  in particles if p.alive]
        effects   = [ef for ef in effects   if ef.alive]

        # Wave complete
        if not wave_queue and not any(e.alive for e in enemies) and not between_waves:
            between_waves = True
            wave_start_t  = now
            wave          += 1

        if castle_hp <= 0:
            castle_hp = 0; game_over = True; highscore = max(highscore, score)

        # ── Draw ──────────────────────────────────────────────────────────────
        for ef in effects:  ef.draw(canvas)
        for e  in enemies:  e.draw(canvas)
        for p  in particles: p.draw(canvas)

        draw_castle(canvas, castle_hp, CASTLE_HP_MAX)
        if hand_found:
            draw_cursor(canvas, palm_x, palm_y, cur_gesture, charge)
        draw_spell_bar(canvas, cur_gesture, charge, max(0, cooldown_until-now))
        draw_top_hud(canvas, score, wave, combo_text, combo_t, now)
        draw_wave_banner(canvas, wave, wave_banner_t, now)
        if between_waves and wave > 0 and now-wave_start_t < 3.0:
            draw_between_waves(canvas, wave, 3.0-(now-wave_start_t))
        draw_pip(canvas, cam)

        cv2.imshow("Spell Caster", canvas)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
