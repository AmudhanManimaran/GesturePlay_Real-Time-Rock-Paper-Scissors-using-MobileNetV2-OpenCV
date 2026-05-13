# utils.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from symbolic_engine import (
    neuro_symbolic_predict,
    decide_winner_symbolic,
    validate_game_state,
    SymbolicStrategyAgent,
)


# ─── Model ────────────────────────────────────────────────────────────────────

def load_rps_model(model_path: str):
    return load_model(model_path)


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    img = cv2.resize(roi, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ─── Neuro-Symbolic Perception ────────────────────────────────────────────────

def predict_class(
    model,
    processed_img: np.ndarray,
    roi: np.ndarray,
    class_names: list,
) -> tuple[str, float, str]:
    """
    Wraps the neuro-symbolic perception gate.
    Returns (gesture, confidence, source).
    """
    return neuro_symbolic_predict(model, processed_img, roi, class_names)


# ─── Game Logic ───────────────────────────────────────────────────────────────

def decide_winner(user: str, bot: str) -> str:
    return decide_winner_symbolic(user, bot)


# ─── HUD Rendering ───────────────────────────────────────────────────────────

def draw_game_info(
    frame: np.ndarray,
    user_move: str,
    comp_move: str,
    result: str,
    user_score: int,
    comp_score: int,
    confidence: float = 1.0,
    source: str = "neural",
    bot_reasoning: str = "",
) -> None:
    """
    Renders the game HUD with neuro-symbolic transparency:
    shows perception source and bot reasoning mode on screen.
    """
    # Perception source colour: green = neural, orange = symbolic fallback
    src_color = (0, 255, 0) if source == "neural" else (0, 165, 255)
    src_label = f"Perception: {source.upper()}  ({confidence:.2f})"

    cv2.putText(frame, f"You: {user_move}",
                (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Bot: {comp_move}",
                (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Result: {result}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Score - You: {user_score}  Bot: {comp_score}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(frame, src_label,
                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, src_color, 2)
    if bot_reasoning:
        cv2.putText(frame, f"Bot strategy: {bot_reasoning}",
                    (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
