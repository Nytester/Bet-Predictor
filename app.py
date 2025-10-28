import os
import pandas as pd
import streamlit as st
from joblib import load
from utils.odds_utils import remove_vig_two_way, expected_value_hold, break_even_cashout, blend_probabilities, recommendation_from_ev, safe_float
from utils.feature_engineering import build_feature_frame
from utils.visualization import gauge_probability, bar_ev

st.set_page_config(page_title="BetIQ — Cash-Out Advisor (NFL)", page_icon="🏈", layout="centered")
st.title("🏈 BetIQ — American Football Cash-Out Advisor")
st.caption("Educational tool: compares EV of holding vs cashing out using market odds + a learned model.")

# --- Sidebar: Model management ---
st.sidebar.header("Model")
model_path = "model/predictor.pkl"
model = None
if os.path.exists(model_path):
    try:
        model = load(model_path)
        st.sidebar.success("Model loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
else:
    st.sidebar.warning("No trained model found. Train one via `model/train_model.py`.")

st.sidebar.markdown("---")
st.sidebar.header("Probability Blend")
alpha = st.sidebar.slider("Weight on ML model (0 = market only, 1 = model only)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("No-Trade Band")
buffer_pct = st.sidebar.slider("Recommendation buffer (%)", 0.0, 10.0, 2.0, 0.5) / 100.0

# --- Tabs ---
tab_live, tab_upload, tab_about = st.tabs(["Live Decision", "Train / Retrain", "About"])

with tab_live:
    st.subheader("1) Your Original Bet")
    c1, c2, c3 = st.columns(3)
    stake = c1.number_input("Stake ($)", min_value=0.0, value=100.0, step=1.0)
    orig_odds = c2.number_input("Original decimal odds", min_value=1.01, value=1.85, step=0.01, format="%.2f")
    cashout = c3.number_input("Current cash-out offer ($)", min_value=0.0, value=95.0, step=1.0)

    st.subheader("2) Market (Live Two-Way Odds)")
    c4, c5 = st.columns(2)
    live_odds_team = c4.number_input("Live odds (your side)", min_value=1.01, value=1.90, step=0.01, format="%.2f")
    live_odds_opp  = c5.number_input("Live odds (opponent)", min_value=1.01, value=2.00, step=0.01, format="%.2f")

    p_market, _ = remove_vig_two_way(live_odds_team, live_odds_opp)

    st.subheader("3) Game Situation (for ML features)")
    c6, c7, c8 = st.columns(3)
    quarter = c6.number_input("Quarter (1-4)", min_value=1, max_value=4, value=3, step=1)
    time_left = c7.text_input("Time left in quarter (MM:SS)", value="08:30")
    score_diff = c8.number_input("Score diff (your team − opp)", value=3, step=1)

    c9, c10, c11 = st.columns(3)
    turnovers = c9.number_input("Your turnovers", min_value=0, value=1, step=1)
    possession = c10.selectbox("Possession", options=["Opponent", "Your team"], index=1)
    yards_gain = c11.number_input("Total yards gained (your team)", min_value=0, value=275, step=5)

    # Build a single-row DF for features
    live_df = pd.DataFrame([{
        "quarter": quarter,
        "time_left": time_left,
        "score_diff": score_diff,
        "odds_pre": orig_odds,
        "odds_live": live_odds_team,
        "turnovers": turnovers,
        "possession": 1 if possession == "Your team" else 0,
        "yards_gain": yards_gain
    }])
    X_live = build_feature_frame(live_df, training=False)

    p_model = None
    if model is not None:
        try:
            p_model = float(model.predict_proba(X_live)[:, 1][0])
        except Exception as e:
            st.warning(f"Model prediction failed: {e}")

    p_blend = blend_probabilities(p_model, p_market, alpha=alpha)

    # --- Results ---
    st.markdown("---")
    st.subheader("Result")

    ev_hold = expected_value_hold(p_blend, stake, orig_odds)
    be_cashout = break_even_cashout(p_blend, stake, orig_odds)
    label, detail = recommendation_from_ev(ev_hold, cashout, buffer_pct=buffer_pct)

    c_top1, c_top2 = st.columns(2)
    with c_top1:
        st.metric("Win Probability (blended)", f"{p_blend*100:.1f} %", help=f"Blend = {alpha:.2f}×Model + {(1-alpha):.2f}×Market")
        st.metric("EV if HOLD", f"${ev_hold:,.2f}")
    with c_top2:
        st.metric("Cash-Out Now", f"${cashout:,.2f}")
        st.metric("Break-even Cash-Out", f"${be_cashout:,.2f}")

    if label == "HOLD":
        st.success(f"✅ Recommendation: {label} — {detail}")
    elif label == "CASH OUT":
        st.warning(f"💼 Recommendation: {label} — {detail}")
    else:
        st.info(f"🤝 Recommendation: {label} — {detail}")

    st.plotly_chart(gauge_probability(p_blend), use_container_width=True)
    st.plotly_chart(bar_ev(ev_hold, cashout), use_container_width=True)

    # Show components
    with st.expander("Sources of probability"):
        st.write({
            "p_market (vig-removed)": round(p_market, 4),
            "p_model (ML)": None if p_model is None else round(p_model, 4),
            "alpha (model weight)": round(alpha, 2),
            "p_blend": round(p_blend, 4)
        })

with tab_upload:
    st.subheader("Train / Retrain Model")
    st.write("Upload a CSV with columns like:")
    st.code("quarter,time_left,score_diff,odds_pre,odds_live,turnovers,possession,yards_gain,result")
    st.caption("result must be Win/Lose (or 1/0). possession: 1=your team, 0=opponent.")

    uploaded = st.file_uploader("Upload past games CSV", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.write("Preview:", df_up.head())
        # Quick train in-app (lightweight)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        from joblib import dump
        from utils.feature_engineering import build_feature_frame, extract_target

        try:
            X = build_feature_frame(df_up, training=True)
            y = extract_target(df_up)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=42
            )
            clf.fit(Xtr, ytr)
            auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
            st.success(f"Model trained. ROC AUC = {auc:.3f}")
            os.makedirs("model", exist_ok=True)
            dump(clf, "model/predictor.pkl")
            st.info("Saved model to model/predictor.pkl. Click 'R' to reload the app and use the new model.")
        except Exception as e:
            st.error(f"Training failed: {e}")

with tab_about:
    st.markdown("""
**How this works**
- Market: We take both sides’ live odds and remove the vig to estimate a fair probability.
- Model: Trained on your historical data to capture context (score/time/possession/etc.).
- Blend: You choose how much to trust the model vs. market.
- Decision: We compare EV(hold) vs. cash-out with a small “no-trade” band.

*Note:* This is not financial advice. Use responsibly.
""")
