import re
import pandas as pd
import numpy as np

TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")

def mmss_to_seconds_left(mmss: str) -> int:
    """
    Convert 'MM:SS' remaining in quarter to seconds remaining in the game (NFL 4x15min).
    """
    if not isinstance(mmss, str):
        return np.nan
    m = TIME_RE.match(mmss)
    if not m:
        return np.nan
    mm = int(m.group(1))
    ss = int(m.group(2))
    sec_in_quarter = 15 * 60
    quarter_elapsed = (15*60 - (mm * 60 + ss))
    return quarter_elapsed  # seconds elapsed in current quarter (we’ll use features relative to quarter)

def seconds_left_game(quarter: int, time_left_mmss: str) -> int:
    """
    Total seconds left in game from start (4 quarters, no OT modeling).
    """
    sec_in_quarter = 15 * 60
    q = int(quarter)
    # seconds remaining in current quarter:
    m = TIME_RE.match(time_left_mmss) if isinstance(time_left_mmss, str) else None
    if not m:
        return np.nan
    mm = int(m.group(1)); ss = int(m.group(2))
    sec_left_in_q = mm*60 + ss
    quarters_left_after_current = max(0, 4 - q)
    return sec_left_in_q + quarters_left_after_current * sec_in_quarter

def build_feature_frame(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Turns raw columns into numeric features for ML.
    Expected columns:
      quarter, time_left, score_diff, odds_pre, odds_live,
      turnovers, possession, yards_gain
    training=True allows missing columns; absent ones are filled with defaults.
    """
    work = df.copy()

    # Fill missing expected columns
    defaults = {
        "quarter": 1, "time_left": "15:00", "score_diff": 0, "odds_pre": 2.0, "odds_live": 2.0,
        "turnovers": 0, "possession": 0, "yards_gain": 0
    }
    for c, d in defaults.items():
        if c not in work.columns:
            work[c] = d

    work["quarter"] = pd.to_numeric(work["quarter"], errors="coerce").fillna(1).clip(1, 4)
    work["score_diff"] = pd.to_numeric(work["score_diff"], errors="coerce").fillna(0)
    work["odds_pre"] = pd.to_numeric(work["odds_pre"], errors="coerce").fillna(2.0).clip(lower=1.01)
    work["odds_live"] = pd.to_numeric(work["odds_live"], errors="coerce").fillna(2.0).clip(lower=1.01)
    work["turnovers"] = pd.to_numeric(work["turnovers"], errors="coerce").fillna(0).clip(lower=0)
    work["possession"] = pd.to_numeric(work["possession"], errors="coerce").fillna(0).clip(0, 1)  # 1 = your team has ball
    work["yards_gain"] = pd.to_numeric(work["yards_gain"], errors="coerce").fillna(0)

    # Time features
    work["seconds_left_game"] = work.apply(
        lambda r: seconds_left_game(int(r["quarter"]), str(r["time_left"])) if pd.notna(r["time_left"]) else np.nan, axis=1
    )
    # Normalize some interactions
    work["lead_per_min_left"] = work["score_diff"] / (work["seconds_left_game"].replace(0, 1) / 60.0)
    work["turnovers_diff"] = work["turnovers"]  # if only own turnovers available

    # Final feature set
    feats = work[[
        "quarter", "score_diff", "odds_pre", "odds_live", "turnovers", "possession",
        "yards_gain", "seconds_left_game", "lead_per_min_left", "turnovers_diff"
    ]].copy()

    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats

def extract_target(df: pd.DataFrame) -> pd.Series:
    """
    Expects a 'result' column with 'Win'/'Lose' (or 1/0).
    """
    if "result" not in df.columns:
        raise ValueError("Training data must include a 'result' column.")
    y = df["result"].astype(str).str.strip().str.lower().map({"win": 1, "lose": 0, "1": 1, "0": 0})
    if y.isna().any():
        # default unknowns to 0
        y = y.fillna(0)
    return y.astype(int)
