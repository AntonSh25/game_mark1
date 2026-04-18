# AI Video Games — Hand Gesture Controller

Two hand-tracking mini-games built with **MediaPipe** and **OpenCV**. No hardware required — just a webcam and your hands.

---

## Games

### 🍉 Fruit Ninja (`fruit_ninja.py`)
Slice flying fruits with your open hand. Avoid bombs. Combo multipliers for slicing multiple fruits in a row.

### 🔮 Spell Caster (`spell_game.py`)
Defend your castle against waves of enemies by casting spells with hand gestures. Combine spells for powerful combos.

---

## Requirements

- Python 3.10+
- Webcam

```bash
pip install mediapipe opencv-python pyautogui numpy
```

The MediaPipe hand landmark model (`hand_landmarker.task`, ~8 MB) is downloaded automatically on first run.

---

## Controls

### Gesture → Spell (Spell Caster)

| Gesture | Spell | Effect |
|---|---|---|
| ✊ Fist | **Fireball** | AoE blast at cursor position |
| ☝️ One finger | **Lightning** | Instant beam across cursor height |
| 🖐 Open hand | **Frost Nova** | AoE freeze at cursor |
| ✌️ Victory | **Wind Slash** | Knockback wave |
| 🤌 Pinch | **Vortex** | Pulls enemies toward cursor for 2 s |

Hold the gesture for ~0.6 s to charge and fire. Your palm position is the aim cursor.

### Combos

| Sequence | Combo | Effect |
|---|---|---|
| Vortex → Fireball | **INFERNO** | Giant explosion |
| Frost Nova → Lightning | **SHATTER** | Instantly kills frozen enemies |
| Fireball → Fireball | **METEOR** | Massive AoE, 8 damage |
| Wind → Vortex | **TORNADO** | Extreme knockback |

### Enemies

| Enemy | HP | Notes |
|---|---|---|
| Goblin | 2 | Fast, no immunity |
| Orc | 5 | Slow, tanky |
| Fire Golem | 7 | Immune to Fireball, weak to Frost/Lightning |
| Ice Wraith | 4 | Immune to Frost, weak to Fireball |
| Shaman | 3 | Heals nearby allies — priority target |

### Gesture → Slice (Fruit Ninja)
Move your open hand fast across a fruit to slice it. Slicing a bomb costs a life.

---

## Running

```bash
# Gesture controller (hand → keyboard/mouse)
python main.py

# Fruit Ninja
python fruit_ninja.py

# Spell Caster
python spell_game.py
```

**First run:** macOS will ask for camera access — grant it in System Settings → Privacy & Security → Camera.

---

## Key Bindings

| Key | Action |
|---|---|
| `Q` | Quit |
| `1` | View mode (main.py) |
| `2` | Action mode — gestures trigger keystrokes (main.py) |
