"""
Spell Caster — defend your castle with hand gestures.

Static spells (hold gesture ~0.6 s):
  fist      → Fireball    (AoE blast at cursor)
  1 finger  → Lightning   (instant beam across cursor height)
  open hand → Frost Nova  (AoE freeze at cursor)
  victory   → Wind Slash  (knockback wave)
  pinch     → Vortex      (2 s pull, setup for combos)

Motion spells (gesture + movement):
  fist   + horizontal swipe  → Meteor Sweep  (fire wave across all enemies)
  palm   + fast push DOWN    → Earthquake    (damages ALL enemies on screen)
  finger + draw CIRCLE       → Arcane Orb    (huge blast at circle center)
  pinch  + pull LEFT, throw RIGHT → Arcane Arrow (single target, max damage)

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
C_SWEEP  = (20,  60, 230)   # Meteor Sweep
C_QUAKE  = (40, 160, 100)   # Earthquake
C_ORB    = (200, 80, 255)   # Arcane Orb
C_ARROW  = (220, 240, 255)  # Arcane Arrow

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
        elif k=="sweep":
            # Horizontal fire wave moving right
            prog = self.t / self.dur
            wx = int(getattr(self,"start_x",0) + prog * (W - getattr(self,"start_x",0)))
            th = max(1, int(18*a))
            cv2.line(f,(wx-30,int(self.y)),(wx+30,int(self.y)),col,th,cv2.LINE_AA)
            cv2.line(f,(wx-30,int(self.y)),(wx+30,int(self.y)),(255,255,255),max(1,th//3),cv2.LINE_AA)
            # Trailing glow
            for i in range(4):
                tx = wx - i*35
                ta = a * (1.0-i*0.22)
                tc = tuple(int(c*ta) for c in self.color)
                cv2.line(f,(tx-20,int(self.y)),(tx+20,int(self.y)),tc,max(1,th-i*3),cv2.LINE_AA)
        elif k=="quake":
            # Expanding ground crack rings
            for ring in range(4):
                r = int((W*0.55) * min(1.0, self.t/self.dur) * (0.3+ring*0.25))
                ry = max(1, r//4)
                cv2.ellipse(f,(int(self.x),int(self.y)),(max(1,r),max(1,ry)),0,0,360,col,2,cv2.LINE_AA)
            # Flash
            if self.t < 0.12:
                fa = 1.0 - self.t/0.12
                arect(f, 0, 0, W, H, (20,60,30), fa*0.35)
        elif k=="orb":
            # Large glowing sphere that expands then contracts
            max_r = getattr(self,"radius",180)
            if self.t < self.dur*0.4:
                r = int(max_r * self.t/(self.dur*0.4))
            else:
                r = int(max_r * (1.0-(self.t-self.dur*0.4)/(self.dur*0.6)))
            r = max(1, r)
            cv2.circle(f,(int(self.x),int(self.y)),r,col,-1,cv2.LINE_AA)
            cv2.circle(f,(int(self.x),int(self.y)),r,(255,255,255),3,cv2.LINE_AA)
            cv2.circle(f,(int(self.x),int(self.y)),max(1,r//2),(255,255,255),1,cv2.LINE_AA)
        elif k=="arrow":
            # Bright bolt from start_x to target
            sx = int(getattr(self,"start_x", int(self.x)))
            tx = int(getattr(self,"target_x", W))
            ty = int(self.y)
            prog = min(1.0, self.t / (self.dur*0.3))
            ex = int(sx + (tx-sx)*prog)
            cv2.line(f,(sx,ty),(ex,ty),col,6,cv2.LINE_AA)
            cv2.line(f,(sx,ty),(ex,ty),(255,255,255),2,cv2.LINE_AA)
            if self.t > self.dur*0.3:
                burst_col = col
                cv2.circle(f,(tx,ty),max(1,int(50*a)),burst_col,-1,cv2.LINE_AA)
                cv2.circle(f,(tx,ty),max(1,int(55*a)),(255,255,255),2,cv2.LINE_AA)

# ── Spell logic ───────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "blast":"fireball","beam":"lightning","nova":"frost","wave":"wind",
    "vortex":"vortex","inferno":"fireball","shatter":"frost","meteor":"fireball","tornado":"wind",
    "sweep":"fireball","quake":"wind","orb":"lightning","arrow":"lightning",
}

# ── Motion spell detector ─────────────────────────────────────────────────────

class MotionDetector:
    SWIPE_PX   = 155   # horizontal px in 0.5 s  → Meteor Sweep
    QUAKE_PY   = 115   # downward px in 0.4 s    → Earthquake
    ORB_DEG    = 300   # cumulative degrees       → Arcane Orb
    ORB_MIN_R  = 38    # min circle radius px
    ARROW_PULL = 65    # pull-left px before charging
    ARROW_VX   = 35    # rightward px/frame to release

    def __init__(self):
        self._pos   = deque(maxlen=45)   # (x, y, t)
        self._gest  = None
        # Orb state
        self._orb_cum   = 0.0
        self._orb_prev  = None
        # Arrow state: "idle" | "pulling" | "charged"
        self._arrow_st  = "idle"
        self._arrow_x0  = 0

    def _reset_gesture(self, g):
        self._pos.clear()
        self._orb_cum  = 0.0
        self._orb_prev = None
        if g != "pinch":
            self._arrow_st = "idle"
        self._gest = g

    def update(self, x, y, gesture, now):
        """Returns (motion_spell_name, cx, cy, extra) or None."""
        if gesture != self._gest:
            self._reset_gesture(gesture)
        self._pos.append((x, y, now))

        if not gesture or len(self._pos) < 3:
            return None

        pos = list(self._pos)

        # ── Meteor Sweep: fist + horizontal swipe ──────────────────────────
        if gesture == "fist":
            recent = [p for p in pos if now - p[2] < 0.5]
            if len(recent) >= 6:
                dx = recent[-1][0] - recent[0][0]
                dy_span = max(p[1] for p in recent) - min(p[1] for p in recent)
                if abs(dx) > self.SWIPE_PX and dy_span < 90:
                    self._reset_gesture(gesture)
                    return ("sweep", x, y, "right" if dx > 0 else "left")

        # ── Earthquake: open hand + fast push down ─────────────────────────
        elif gesture == "open":
            recent = [p for p in pos if now - p[2] < 0.4]
            if len(recent) >= 5:
                dy = recent[-1][1] - recent[0][1]
                dx_span = max(p[0] for p in recent) - min(p[0] for p in recent)
                if dy > self.QUAKE_PY and dx_span < 110:
                    self._reset_gesture(gesture)
                    return ("quake", x, y, None)

        # ── Arcane Orb: point finger + draw circle ─────────────────────────
        elif gesture == "point":
            vx = pos[-1][0] - pos[-2][0]
            vy = pos[-1][1] - pos[-2][1]
            if abs(vx) > 1 or abs(vy) > 1:
                angle = math.atan2(vy, vx)
                if self._orb_prev is not None:
                    da = angle - self._orb_prev
                    while da >  math.pi: da -= 2*math.pi
                    while da < -math.pi: da += 2*math.pi
                    self._orb_cum += da
                self._orb_prev = angle

            if abs(self._orb_cum) >= math.radians(self.ORB_DEG):
                cx = int(sum(p[0] for p in pos) / len(pos))
                cy = int(sum(p[1] for p in pos) / len(pos))
                radius = sum(math.hypot(p[0]-cx, p[1]-cy) for p in pos) / len(pos)
                if radius >= self.ORB_MIN_R:
                    self._reset_gesture(gesture)
                    return ("orb", cx, cy, int(radius))

        # ── Arcane Arrow: pinch + pull left + throw right ──────────────────
        elif gesture == "pinch":
            if self._arrow_st == "idle":
                self._arrow_x0 = x
                self._arrow_st = "pulling"
            elif self._arrow_st == "pulling":
                if self._arrow_x0 - x > self.ARROW_PULL:   # moved left = pulling
                    self._arrow_st = "charged"
            elif self._arrow_st == "charged":
                vx = pos[-1][0] - pos[-3][0]               # px over 2 frames
                if vx > self.ARROW_VX:                      # fast throw right
                    self._arrow_st = "idle"
                    self._reset_gesture(gesture)
                    return ("arrow", x, y, None)

        return None

    def orb_progress(self):
        return min(1.0, abs(self._orb_cum) / math.radians(self.ORB_DEG))

    def orb_trail(self):
        return [(p[0], p[1]) for p in self._pos] if self._gest == "point" else []

    def arrow_state(self):
        return self._arrow_st

    def arrow_charge_x(self):
        return self._arrow_x0

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

def cast_motion(spell, cx, cy, extra, enemies, particles, effects):
    """Fire a motion spell. Returns score earned."""
    earned = 0

    if spell == "sweep":
        # Horizontal fire wave — damages all enemies near palm_y
        radius_y = 70
        for e in enemies:
            if not e.alive: continue
            if abs(e.y - cy) < radius_y + e.h // 2:
                e.take_damage(4, "fireball")
                burst(particles, e.x, e.y, C_SWEEP, 14, 6)
                if not e.alive and not e.scored:
                    earned += e.pts; e.scored = True
        effects.append(Effect("sweep", cx, cy, C_SWEEP, 0.7, start_x=cx))

    elif spell == "quake":
        # Hits ALL enemies regardless of position
        for e in enemies:
            if not e.alive: continue
            e.take_damage(2, "wind")
            e.frozen = 0.8
            burst(particles, e.x, e.y, C_QUAKE, 10, 4)
            if not e.alive and not e.scored:
                earned += e.pts; e.scored = True
        effects.append(Effect("quake", W//2, H-60, C_QUAKE, 0.8))

    elif spell == "orb":
        radius = max(100, extra or 120)
        for e in enemies:
            if not e.alive: continue
            if math.hypot(e.x-cx, e.y-cy) < radius + e.w:
                e.take_damage(6, "lightning")
                burst(particles, e.x, e.y, C_ORB, 16, 7)
                if not e.alive and not e.scored:
                    earned += e.pts; e.scored = True
        effects.append(Effect("orb", cx, cy, C_ORB, 0.7, radius=radius))
        burst(particles, cx, cy, C_ORB, 30, 10)

    elif spell == "arrow":
        # Find nearest enemy, deal massive damage
        target = min(
            (e for e in enemies if e.alive),
            key=lambda e: math.hypot(e.x-cx, e.y-cy),
            default=None,
        )
        tx = target.x if target else W
        if target:
            target.take_damage(12, "lightning")
            burst(particles, target.x, target.y, C_ARROW, 20, 8)
            if not target.alive and not target.scored:
                earned += target.pts; target.scored = True
        effects.append(Effect("arrow", cx, cy, C_ARROW, 0.55,
                               start_x=cx, target_x=int(tx)))

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

def draw_cursor(frame, x, y, gesture, motion_det):
    # ── Arcane Orb: draw circle trail ──────────────────────────────────────
    if gesture == "point":
        trail = motion_det.orb_trail()
        prog  = motion_det.orb_progress()
        if len(trail) >= 2:
            for i in range(1, len(trail)):
                a = i / len(trail)
                col = tuple(int(c*a) for c in C_ORB)
                cv2.line(frame, trail[i-1], trail[i], col, max(1,int(3*a)), cv2.LINE_AA)
        if prog > 0:
            cv2.ellipse(frame,(x,y),(22,22),-90,0,int(360*prog),C_ORB,3,cv2.LINE_AA)
            put(frame, f"ORB {int(prog*100)}%", (x+26, y+6), 0.45, C_ORB, 1)

    # ── Arcane Arrow: show bow-draw indicator ──────────────────────────────
    elif gesture == "pinch":
        st = motion_det.arrow_state()
        if st == "charged":
            ax0 = motion_det.arrow_charge_x()
            cv2.line(frame,(x,y),(ax0,y),C_ARROW,3,cv2.LINE_AA)
            cv2.line(frame,(ax0,y-20),(ax0,y+20),C_ARROW,2,cv2.LINE_AA)
            put(frame,"CHARGED! throw ->",(x-60,y-28),0.5,C_ARROW,1)
        elif st == "pulling":
            ax0 = motion_det.arrow_charge_x()
            pct = min(1.0, max(0, ax0-x) / MotionDetector.ARROW_PULL)
            put(frame,f"<- pull {int(pct*100)}%",(x-60,y-28),0.45,C_ARROW,1)

    # ── Earthquake: show downward arrow ───────────────────────────────────
    elif gesture == "open":
        cv2.arrowedLine(frame,(x,y-10),(x,y+30),C_QUAKE,2,cv2.LINE_AA,tipLength=0.4)
        put(frame,"push DOWN",(x+18,y+10),0.42,C_QUAKE,1)

    # ── Meteor Sweep: show horizontal arrow ───────────────────────────────
    elif gesture == "fist":
        cv2.arrowedLine(frame,(x-30,y),(x+30,y),C_SWEEP,2,cv2.LINE_AA,tipLength=0.35)
        put(frame,"SWIPE",(x-16,y-18),0.42,C_SWEEP,1)

    # Basic cursor dot
    cv2.circle(frame,(x,y),12,(70,60,90),1,cv2.LINE_AA)
    cv2.circle(frame,(x,y),3,(110,90,130),-1,cv2.LINE_AA)

def draw_spell_bar(frame):
    y0 = H - 72
    arect(frame, 0, y0-4, W, H, (8,5,14), 0.82)
    motion_hints = [
        (C_SWEEP,  "fist + SWIPE",       "Meteor Sweep"),
        (C_QUAKE,  "palm + push DOWN",   "Earthquake"),
        (C_ORB,    "finger + CIRCLE",    "Arcane Orb"),
        (C_ARROW,  "pinch: pull<-  ->",  "Arcane Arrow"),
    ]
    bw = (W - 20) // 4
    for i, (col, hint, name) in enumerate(motion_hints):
        bx = 10 + i * bw
        cv2.rectangle(frame, (bx, y0), (bx+bw-4, y0+64), (25,18,38), -1)
        cv2.rectangle(frame, (bx, y0), (bx+bw-4, y0+64), col, 1)
        put(frame, name, (bx+6, y0+20), 0.5, col, 1)
        put(frame, hint, (bx+6, y0+42), 0.38, (110,95,140), 1)

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

def draw_hand_icon(img, cx, cy, gesture, color=(170, 150, 210), s=1.0):
    """Draw a simplified hand silhouette for the given gesture."""
    # Which fingers are extended: [thumb, index, middle, ring, pinky]
    UP = {
        "fist":    [0, 0, 0, 0, 0],
        "open":    [1, 1, 1, 1, 1],
        "point":   [0, 1, 0, 0, 0],
        "victory": [0, 1, 1, 0, 0],
        "pinch":   [0, 0, 0, 0, 0],
        # motion spells reuse base gestures
        "sweep":   [0, 0, 0, 0, 0],
        "quake":   [1, 1, 1, 1, 1],
        "orb":     [0, 1, 0, 0, 0],
        "arrow":   [0, 0, 0, 0, 0],
    }
    up = UP.get(gesture, [0, 0, 0, 0, 0])

    pw, ph = int(28*s), int(22*s)
    palm_y = cy + int(10*s)
    # Palm
    cv2.rectangle(img, (cx-pw//2, palm_y-ph//2), (cx+pw//2, palm_y+ph//2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx-pw//2, palm_y-ph//2), (cx+pw//2, palm_y+ph//2), (0,0,0), 1, cv2.LINE_AA)

    # Four fingers (index–pinky)
    finger_xs = [-int(13*s), -int(4*s), int(5*s), int(14*s)]
    fw = max(2, int(7*s))
    for i, fx in enumerate(finger_xs):
        fh = int(26*s) if up[i+1] else int(10*s)
        fy_top = palm_y - ph//2 - fh
        cv2.rectangle(img, (cx+fx-fw//2, fy_top), (cx+fx+fw//2, palm_y-ph//2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (cx+fx-fw//2, fy_top), (cx+fx+fw//2, palm_y-ph//2), (0,0,0), 1, cv2.LINE_AA)

    # Thumb (horizontal, left side)
    tw = int(18*s) if up[0] else int(8*s)
    tx = cx - pw//2 - tw
    ty_c = palm_y - int(2*s)
    cv2.rectangle(img, (tx, ty_c - max(2,int(5*s))), (cx-pw//2, ty_c + max(2,int(5*s))), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (tx, ty_c - max(2,int(5*s))), (cx-pw//2, ty_c + max(2,int(5*s))), (0,0,0), 1, cv2.LINE_AA)

    # Pinch: circle between thumb and index tip
    if gesture == "pinch" or gesture == "arrow":
        tip_x = cx - pw//2 - int(2*s)
        tip_y = palm_y - ph//2 - int(8*s)
        cv2.circle(img, (tip_x, tip_y), max(2, int(7*s)), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx-pw//2, palm_y-ph//2-int(10*s)), (tip_x, tip_y), color, max(1,int(2*s)), cv2.LINE_AA)


def draw_tutorial(frame):
    """Full tutorial overlay — motion spells only."""
    arect(frame, 0, 0, W, H, (5, 3, 12), 0.93)

    put(frame, "HOW TO CAST SPELLS", (W//2 - 230, 48), 1.4, (200, 180, 255), 3)
    cv2.line(frame, (60, 62), (W-60, 62), (60, 45, 90), 1)
    put(frame, "GESTURE + MOVEMENT", (W//2 - 155, 92), 0.65, (140, 120, 180), 1)
    cv2.line(frame, (60, 105), (W-60, 105), (50, 38, 72), 1)

    motion_spells = [
        ("fist",  C_SWEEP, "Meteor Sweep", "Make a FIST  +  SWIPE left or right",  "Hits all enemies in the row", "h"),
        ("open",  C_QUAKE, "Earthquake",   "OPEN hand  +  PUSH DOWN fast",          "Damages ALL enemies on screen","d"),
        ("point", C_ORB,   "Arcane Orb",   "Point 1 FINGER  +  Draw a CIRCLE",      "Huge blast at circle center",  "c"),
        ("arrow", C_ARROW, "Arcane Arrow", "PINCH  +  Pull LEFT, then throw RIGHT", "One target, maximum damage",   "lr"),
    ]

    row_h = (H - 180) // 4
    for i, (gest, col, name, motion_hint, desc, sym) in enumerate(motion_spells):
        y = 120 + i * row_h
        icon_x, icon_y = 130, y + row_h // 2 - 10

        draw_hand_icon(frame, icon_x, icon_y, gest, col, s=1.4)

        ax, ay = icon_x + 55, icon_y
        if sym == "h":
            cv2.arrowedLine(frame, (ax-22, ay), (ax+22, ay), col, 2, cv2.LINE_AA, tipLength=0.35)
        elif sym == "d":
            cv2.arrowedLine(frame, (ax, ay-18), (ax, ay+18), col, 2, cv2.LINE_AA, tipLength=0.35)
        elif sym == "c":
            cv2.circle(frame, (ax, ay), 16, col, 2, cv2.LINE_AA)
            cv2.arrowedLine(frame, (ax, ay-16), (ax+8, ay-16), col, 2, cv2.LINE_AA, tipLength=0.6)
        elif sym == "lr":
            cv2.arrowedLine(frame, (ax+12, ay-7), (ax-12, ay-7), col, 2, cv2.LINE_AA, tipLength=0.45)
            cv2.arrowedLine(frame, (ax-12, ay+7), (ax+12, ay+7), col, 2, cv2.LINE_AA, tipLength=0.45)

        put(frame, name,        (220, y + row_h//2 - 22), 0.9,  col, 2)
        put(frame, motion_hint, (220, y + row_h//2 +  4), 0.48, (160, 145, 190), 1)
        put(frame, desc,        (220, y + row_h//2 + 22), 0.4,  (100, 88, 130), 1)

        if i < len(motion_spells)-1:
            cv2.line(frame, (60, y+row_h-4), (W-60, y+row_h-4), (35, 28, 52), 1)

    put(frame, "Press  H  to close", (W//2 - 115, H-18), 0.65, (80, 70, 110), 1)


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
    cooldown_until = 0.0
    combo_text     = "";    combo_t = 0.0
    motion_det     = MotionDetector()

    game_over    = False;  open_frames = 0
    show_tutorial = True   # shown before first wave
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

        # ── Tutorial screen ───────────────────────────────────────────────────
        if show_tutorial:
            draw_castle(canvas, castle_hp, CASTLE_HP_MAX)
            draw_tutorial(canvas)
            draw_pip(canvas, cam)
            cv2.imshow("Spell Caster", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('h'), ord('H'), ord(' '), 13):  # H / Space / Enter
                show_tutorial = False
            elif key == ord('q'):
                break
            continue

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
                cur_gesture=None; cooldown_until=0.0
                combo_text=""; combo_t=0.0; game_over=False; open_frames=0
                motion_det = MotionDetector()
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

        # ── Motion spell detection ────────────────────────────────────────────
        motion_result = motion_det.update(palm_x, palm_y, raw_gesture, now) if hand_found else None
        if motion_result and now >= cooldown_until:
            mspell, mcx, mcy, mextra = motion_result
            pts = cast_motion(mspell, mcx, mcy, mextra, enemies, particles, effects)
            score += pts * 15
            combo_text = mspell.upper().replace("_"," ")
            combo_t    = now
            cooldown_until = now + 0.5

        cur_gesture = raw_gesture

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
            draw_cursor(canvas, palm_x, palm_y, cur_gesture, motion_det)
        draw_spell_bar(canvas)
        draw_top_hud(canvas, score, wave, combo_text, combo_t, now)
        draw_wave_banner(canvas, wave, wave_banner_t, now)
        if between_waves and wave > 0 and now-wave_start_t < 3.0:
            draw_between_waves(canvas, wave, 3.0-(now-wave_start_t))
        draw_pip(canvas, cam)

        if show_tutorial:
            draw_tutorial(canvas)

        cv2.imshow("Spell Caster", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key in (ord('h'), ord('H')):
            show_tutorial = not show_tutorial

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
