import plotly.graph_objects as go

def gauge_probability(p: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(0, min(100, p*100)),
        title={'text': "Win Probability (%)"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def bar_ev(ev_hold: float, cashout: float):
    fig = go.Figure(data=[
        go.Bar(name="EV Hold", x=["EV"], y=[ev_hold]),
        go.Bar(name="Cash Out", x=["EV"], y=[cashout]),
    ])
    fig.update_layout(barmode='group', height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig
