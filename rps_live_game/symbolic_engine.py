# symbolic_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Symbolic Reasoning Layer for GesturePlay Neuro-Symbolic Pipeline
#
# Responsibilities:
#   1. Symbolic Fallback Classifier  – rule-based gesture recognition via
#      OpenCV contour/hull analysis, invoked when neural confidence < threshold.
#   2. Symbolic Strategy Agent       – logic-driven bot that tracks player
#      history, infers dominant patterns, and counters deterministically.
#   3. Constraint Validator          – ensures every game state is legal before
#      it is committed (guards against corrupt neural outputs).
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from collections import Counter, deque


# ─── Constants ────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.75          # below this → symbolic fallback fires
VALID_GESTURES       = {"rock", "paper", "scissors"}
BEATS                = {             # symbolic game-logic knowledge base
    "rock":     "scissors",
    "scissors": "paper",
    "paper":    "rock",
}
COUNTER_MOVE = {v: k for k, v in BEATS.items()}   # what beats each gesture
HISTORY_WINDOW = 10                  # rolling window for pattern inference


# ─── 1. Symbolic Fallback Classifier ─────────────────────────────────────────

def symbolic_gesture_classify(roi: np.ndarray) -> str:
    """
    Rule-based gesture classifier using contour geometry and convex hull
    defect analysis.  Called only when neural confidence < CONFIDENCE_THRESHOLD.

    Heuristic rules (derived from hand-shape geometry):
      • Very low solidity  → open/spread hand  → paper
      • High defect count  → multiple finger gaps → scissors / rock ambiguity
        resolved by aspect ratio of bounding rect
      • High solidity + compact shape → closed fist → rock
    """
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "rock"   # safe symbolic default

    cnt      = max(contours, key=cv2.contourArea)
    area     = cv2.contourArea(cnt)
    hull     = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return "rock"

    solidity = area / hull_area                     # [0, 1] — compactness ratio

    # Count convexity defects (finger gaps)
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    defects      = cv2.convexityDefects(cnt, hull_indices)
    significant_defects = 0
    if defects is not None:
        for defect in defects:
            _, _, _, depth = defect[0]
            if depth / 256.0 > 20:                  # depth threshold in pixels
                significant_defects += 1

    # Symbolic rule set
    if solidity > 0.90:
        return "rock"                               # compact fist
    elif significant_defects >= 2:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 1.0
        if aspect > 0.85:
            return "paper"                          # wide spread
        return "scissors"                           # two-finger gap
    else:
        return "paper"                              # default open-hand


# ─── 2. Neuro-Symbolic Perception Gate ───────────────────────────────────────

def neuro_symbolic_predict(
    model,
    processed_img: np.ndarray,
    roi: np.ndarray,
    class_names: list,
) -> tuple[str, float, str]:
    """
    Primary perception function.

    Returns
    -------
    gesture    : str   – final resolved gesture
    confidence : float – neural confidence (0–1)
    source     : str   – 'neural' | 'symbolic_fallback'
    """
    raw_pred   = model.predict(processed_img, verbose=0)[0]   # shape (3,)
    confidence = float(np.max(raw_pred))
    neural_gesture = class_names[int(np.argmax(raw_pred))]

    if confidence >= CONFIDENCE_THRESHOLD:
        return neural_gesture, confidence, "neural"

    # Neural uncertain → invoke symbolic fallback
    symbolic_gesture = symbolic_gesture_classify(roi)
    return symbolic_gesture, confidence, "symbolic_fallback"


# ─── 3. Constraint Validator ──────────────────────────────────────────────────

def validate_game_state(user_move: str, bot_move: str) -> bool:
    """
    Formally verifies that both moves belong to the valid gesture domain.
    Rejects any state that violates the game's logical constraints.
    """
    return user_move in VALID_GESTURES and bot_move in VALID_GESTURES


def decide_winner_symbolic(user: str, bot: str) -> str:
    """
    Constraint-enforced outcome resolution.
    Raises ValueError on illegal state — guarantees zero illegal game states.
    """
    if not validate_game_state(user, bot):
        raise ValueError(f"[CONSTRAINT VIOLATION] Illegal state: user={user}, bot={bot}")
    if user == bot:
        return "Draw"
    if BEATS[user] == bot:
        return "You Win"
    return "Bot Wins"


# ─── 4. Symbolic Strategy Agent ───────────────────────────────────────────────

class SymbolicStrategyAgent:
    """
    Logic-driven adversarial agent.

    Strategy (priority order):
      1. Frequency dominance  – if one gesture appears in ≥50 % of the rolling
         window, counter it deterministically.
      2. Transition pattern   – detect if the player tends to cycle gestures
         in a fixed sequence (rock→scissors→paper etc.) and anticipate next.
      3. Fallback             – uniform random (when no pattern is detectable).

    All decisions are made through explicit symbolic inference rules,
    not learned weights.
    """

    def __init__(self, window: int = HISTORY_WINDOW):
        self.history: deque[str] = deque(maxlen=window)
        self.round_number: int   = 0

    def record(self, user_move: str) -> None:
        if user_move in VALID_GESTURES:
            self.history.append(user_move)
            self.round_number += 1

    def _dominant_gesture(self) -> str | None:
        """Rule: if one gesture is ≥50 % of history, it is dominant."""
        if len(self.history) < 3:
            return None
        counts = Counter(self.history)
        most_common, freq = counts.most_common(1)[0]
        if freq / len(self.history) >= 0.50:
            return most_common
        return None

    def _transition_prediction(self) -> str | None:
        """
        Rule: detect the most frequent (last_move → next_move) transition
        in history and predict the next move accordingly.
        """
        if len(self.history) < 4:
            return None
        history_list = list(self.history)
        transitions: Counter = Counter()
        for i in range(len(history_list) - 1):
            transitions[(history_list[i], history_list[i + 1])] += 1
        if not transitions:
            return None
        last_move   = history_list[-1]
        # find the most likely next move given last_move
        candidates  = {nxt: cnt for (prv, nxt), cnt in transitions.items() if prv == last_move}
        if not candidates:
            return None
        predicted_user = max(candidates, key=candidates.get)
        # return what beats the predicted user move
        return COUNTER_MOVE[predicted_user]

    def choose_move(self) -> tuple[str, str]:
        """
        Returns (bot_move, reasoning_label) for display/logging.
        """
        # Rule 1 – dominance counter
        dominant = self._dominant_gesture()
        if dominant:
            return COUNTER_MOVE[dominant], f"counter-dominant:{dominant}"

        # Rule 2 – transition prediction
        transition_counter = self._transition_prediction()
        if transition_counter:
            return transition_counter, "transition-pattern"

        # Rule 3 – fallback random
        import random
        move = random.choice(list(VALID_GESTURES))
        return move, "random-fallback"
