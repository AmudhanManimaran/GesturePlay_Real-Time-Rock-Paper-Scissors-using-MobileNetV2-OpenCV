# GesturePlay — Neuro-Symbolic Visual Logic Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A real-time Rock-Paper-Scissors engine built on a **Neuro-Symbolic AI pipeline** — a fine-tuned **MobileNetV2 CNN** handles visual perception under uncertainty, while a **Symbolic Reasoning Layer** provides confidence-gated fallback classification, logic-driven adversarial strategy, and formal constraint validation — guaranteeing zero illegal game states regardless of neural probabilistic variance.

---

## Why Neuro-Symbolic?

Pure neural systems are powerful but opaque and brittle under low-confidence inputs. Pure rule systems are rigid and cannot handle raw pixel data. GesturePlay combines **both**:

| Layer | Technology | Role |
|---|---|---|
| Neural Perception | MobileNetV2 (fine-tuned) | Gesture recognition from raw frames |
| Symbolic Fallback | OpenCV contour + hull analysis | Rule-based classifier when neural confidence < 0.75 |
| Strategy Agent | `SymbolicStrategyAgent` | Logic-driven bot using frequency & transition rules |
| Constraint Validator | `validate_game_state()` | Formally rejects any illegal game state |

Neither layer alone is sufficient. Together they form a robust, explainable pipeline.

---

## System Architecture

```
Webcam Feed
     │
     ▼
┌──────────────────────────────────────┐
│         ROI Extraction (OpenCV)      │  ← 300×300 hand region crop
└──────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│     MobileNetV2 Neural Classifier    │  ← Fine-tuned, 224×224 input
│     outputs: gesture + confidence    │
└──────────────────────────────────────┘
     │
     ├─── confidence ≥ 0.75 ──────────────────────────┐
     │                                                 │
     └─── confidence < 0.75 ──►  ┌──────────────────────────────────┐
                                  │  Symbolic Fallback Classifier    │
                                  │  (Contour Solidity + Hull Defects│
                                  │   rule-based gesture resolution) │
                                  └──────────────────────────────────┘
                                                       │
     ┌─────────────────────────────────────────────────┘
     │  Resolved Gesture (neural OR symbolic)
     ▼
┌──────────────────────────────────────┐
│      Constraint Validator            │  ← Rejects illegal states formally
│   validate_game_state(user, bot)     │
└──────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│     SymbolicStrategyAgent            │  ← Logic-driven bot
│  Rule 1: Frequency Dominance Counter │     No random guessing
│  Rule 2: Transition Pattern Predict  │     Tracks 10-round history
│  Rule 3: Random Fallback             │
└──────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│  Symbolic Outcome Resolution         │  ← BEATS knowledge base
│  decide_winner_symbolic(user, bot)   │     Raises on invalid state
└──────────────────────────────────────┘
     │
     ▼
  Game Outcome + Score Update + HUD Render
```

---

## Key Features

- **Confidence-gated symbolic fallback** — OpenCV contour solidity and convex hull defect analysis resolves ambiguous gestures the neural model is uncertain about
- **Logic-driven adversarial bot** — `SymbolicStrategyAgent` applies priority-ordered symbolic rules (frequency dominance → transition prediction → random) over a rolling 10-round history window
- **Formal constraint validation** — every `(user_move, bot_move)` pair is verified against the valid gesture domain before the round is committed; illegal states raise `ValueError`
- **Transparent HUD** — live display of perception source (`NEURAL` / `SYMBOLIC_FALLBACK`), neural confidence score, and active bot strategy label
- **Explainable decisions** — every round logs source, confidence, and bot reasoning to console for full auditability

---

## Project Structure

```
GesturePlay/
│
├── rps_live_game/
│   ├── rps_live_predict.py          # Main game loop — neuro-symbolic pipeline orchestration
│   ├── utils.py                     # Perception, preprocessing, HUD rendering
│   ├── symbolic_engine.py           # Symbolic fallback, strategy agent, constraint validator
│   ├── rps_mobilenetv2_final.keras  # Fine-tuned MobileNetV2 weights
│   ├── rps.ipynb                    # Model training notebook (Google Colab)
│   ├── labels.txt                   # Class label definitions
│   └── test_tf.py                   # TensorFlow version check
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Symbolic Engine — `symbolic_engine.py`

### 1. Symbolic Fallback Classifier
Invoked when MobileNetV2 confidence drops below **0.75**. Uses three OpenCV-derived geometric rules:

```
solidity > 0.90                          → rock   (compact closed fist)
significant_defects ≥ 2 + wide aspect   → paper  (spread open hand)
significant_defects ≥ 2 + narrow aspect → scissors (two-finger gap)
default                                  → paper
```

`solidity = contour_area / convex_hull_area` — a measure of hand compactness.

### 2. Symbolic Strategy Agent
Applies rules in strict priority order over a rolling history window:

```
Rule 1 — Frequency Dominance:
  IF any gesture appears in ≥ 50% of last 10 rounds
  THEN play the move that beats it

Rule 2 — Transition Pattern:
  IF a (prev_move → next_move) transition is dominant in history
  THEN predict user's next move and counter it

Rule 3 — Random Fallback:
  No detectable pattern → uniform random
```

### 3. Constraint Validator
```python
def validate_game_state(user_move, bot_move) -> bool:
    return user_move in VALID_GESTURES and bot_move in VALID_GESTURES
```
Every game state is formally checked before outcome resolution. Invalid states are rejected with a `ValueError` — the round is skipped entirely.

---

## Model Performance

| Metric | Value |
|---|---|
| Base Model | MobileNetV2 (ImageNet pre-trained) |
| Input Size | 224 × 224 × 3 |
| Classes | Rock, Paper, Scissors |
| Fine-tuning | Last 30 layers unfrozen |
| Optimizer | Adam (lr=1e-5) |
| Augmentation | Rotation, Zoom, Shift, Horizontal Flip |
| Confidence Threshold | 0.75 (below → symbolic fallback) |

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/AmudhanManimaran/GesturePlay.git
cd GesturePlay/rps_live_game
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
python rps_live_predict.py
```

---

## How to Play

1. Run `rps_live_predict.py`
2. Place your hand inside the **ROI box** on screen
3. Show **Rock**, **Paper**, or **Scissors**
4. Every **5 seconds** the pipeline fires — gesture is classified, validated, and resolved
5. Watch the HUD — green label means neural prediction was confident; orange means symbolic fallback activated
6. Press **`q`** to quit

---

## Console Output (per round)

```
[ROUND] user=rock (neural, conf=0.94) | bot=paper (counter-dominant:rock) | result=Bot Wins
[ROUND] user=scissors (symbolic_fallback, conf=0.61) | bot=rock (transition-pattern) | result=Bot Wins
```

Every round is fully auditable — perception source, confidence, bot reasoning, and outcome are all logged.

---

## Requirements

```
tensorflow>=2.10.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Amudhan Manimaran**
- Portfolio: [amudhanmanimaran.github.io/Portfolio](https://amudhanmanimaran.github.io/Portfolio/)
- LinkedIn: [linkedin.com/in/amudhan-manimaran-3621bb32a](https://www.linkedin.com/in/amudhan-manimaran-3621bb32a)
- GitHub: [github.com/AmudhanManimaran](https://github.com/AmudhanManimaran)
