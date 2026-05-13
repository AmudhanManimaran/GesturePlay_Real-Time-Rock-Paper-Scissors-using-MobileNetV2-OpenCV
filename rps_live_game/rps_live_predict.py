# rps_live_predict.py
# ─────────────────────────────────────────────────────────────────────────────
# GesturePlay — Neuro-Symbolic Visual Logic Engine
# Main game loop
#
# Pipeline (per round):
#   Frame → ROI → MobileNetV2 inference
#       ├─ confidence ≥ 0.75 → neural gesture accepted
#       └─ confidence < 0.75 → symbolic fallback (contour/hull rules)
#   Validated gesture → Constraint Validator → legal state check
#   SymbolicStrategyAgent → pattern-inferred bot move
#   Outcome resolved via symbolic game-logic knowledge base
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import time
from utils import (
    load_rps_model,
    preprocess_roi,
    predict_class,
    draw_game_info,
    decide_winner,
)
from symbolic_engine import SymbolicStrategyAgent, validate_game_state

# ─── Init ─────────────────────────────────────────────────────────────────────

model_path  = "rps_mobilenetv2_final.keras"
model       = load_rps_model(model_path)
class_names = ['paper', 'rock', 'scissors']
agent       = SymbolicStrategyAgent(window=10)

print("[INFO] Model loaded. Neuro-Symbolic pipeline active.")
print("[INFO] Starting webcam. Press 'q' to quit.")

cap = cv2.VideoCapture(0)

user_score     = 0
bot_score      = 0
last_result    = "Waiting..."
user_move      = "None"
bot_move       = "None"
confidence     = 0.0
source         = "neural"
bot_reasoning  = ""
next_play_time = time.time() + 5

# ─── Game Loop ────────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame not captured.")
        break

    # ROI window
    x1, y1, x2, y2 = 50, 50, 350, 350
    roi = frame[y1:y2, x1:x2]

    # ROI border: blue = waiting, green = captured
    current_time = time.time()
    border_color = (0, 255, 0) if current_time >= next_play_time else (255, 0, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

    if current_time >= next_play_time:

        # ── 1. Neural perception + symbolic fallback gate ──
        processed_img             = preprocess_roi(roi)
        user_move, confidence, source = predict_class(
            model, processed_img, roi, class_names
        )

        # ── 2. Constraint validation before committing state ──
        bot_move, bot_reasoning = agent.choose_move()

        if not validate_game_state(user_move, bot_move):
            print(f"[CONSTRAINT VIOLATION] Illegal state detected — skipping round.")
            next_play_time = current_time + 5
            continue

        # ── 3. Symbolic outcome resolution ──
        last_result = decide_winner(user_move, bot_move)

        # ── 4. Update scores and agent history ──
        if last_result == "You Win":
            user_score += 1
        elif last_result == "Bot Wins":
            bot_score += 1

        agent.record(user_move)   # feed symbolic agent for next-round inference

        print(
            f"[ROUND] user={user_move} ({source}, conf={confidence:.2f}) | "
            f"bot={bot_move} ({bot_reasoning}) | result={last_result}"
        )

        next_play_time = current_time + 5

    else:
        remaining   = int(next_play_time - current_time)
        last_result = f"Next round in {remaining}s"

    # ── HUD ──
    draw_game_info(
        frame, user_move, bot_move, last_result,
        user_score, bot_score, confidence, source, bot_reasoning
    )
    cv2.imshow("GesturePlay — Neuro-Symbolic Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Session ended.")
print(f"[FINAL] You: {user_score}  Bot: {bot_score}")
