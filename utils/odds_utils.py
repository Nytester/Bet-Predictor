import math

def decimal_to_implied_prob(odds_decimal: float) -> float:
    if odds_decimal <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    return 1.0 / odds_decimal

def remove_vig_two_way(odds_team: float, odds_opp: float) -> tuple[float, float]:
    """
    Convert two-way decimal odds to fair probabilities by normalizing implied probs.
    Returns (p_team, p_opp) with p_team + p_opp = 1.
    """
    p_team_raw = decimal_to_implied_prob(odds_team)
    p_opp_raw  = decimal_to_implied_prob(odds_opp)
    s = p_team_raw + p_opp_raw
    if s <= 0:
        raise ValueError("Invalid odds pair.")
    return p_team_raw / s, p_opp_raw / s

def expected_value_hold(p_win: float, stake: float, original_decimal_odds: float) -> float:
    """
    EV of holding the original bet. Assumes loss returns 0 (stake already at risk).
    """
    payout_if_win = stake * original_decimal_odds
    return p_win * payout_if_win

def break_even_cashout(p_win: float, stake: float, original_decimal_odds: float) -> float:
    return expected_value_hold(p_win, stake, original_decimal_odds)

def recommendation_from_ev(ev_hold: float, cashout: float, buffer_pct: float = 0.02) -> tuple[str, str]:
    """
    buffer_pct adds a small no-trade band to avoid flip-flopping due to noise.
    Returns (label, detail).
    """
    if cashout <= 0:
        return "HOLD", "Cash-out is zero or invalid."

    edge = ev_hold - cashout
    thresh = max(buffer_pct * cashout, 0.01)
    if edge > thresh:
        return "HOLD", f"EV_hold exceeds cash-out by ${edge:.2f} (> {buffer_pct*100:.0f}% band)."
    if edge < -thresh:
        return "CASH OUT", f"Cash-out exceeds EV_hold by ${-edge:.2f} (> {buffer_pct*100:.0f}% band)."
    return "NO ACTION", "Within no-trade band; decision is close. Consider risk tolerance and bankroll."

def blend_probabilities(p_model: float | None, p_market: float | None, alpha: float) -> float:
    """
    Blend model and market (vig-removed) probabilities:
      p_blend = alpha * p_model + (1 - alpha) * p_market
    If one is None, fall back to the other.
    """
    if p_model is None and p_market is None:
        raise ValueError("Need at least one probability source.")
    if p_model is None:
        return float(p_market)
    if p_market is None:
        return float(p_model)
    return float(alpha) * float(p_model) + (1.0 - float(alpha)) * float(p_market)

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default
