import streamlit as st
import joblib

model = joblib.load("model.pkl")

st.set_page_config(
    page_title="NBA Win Probability",
    page_icon="🏀"
)

st.title("🏀 NBA Win Probability Predictor")

st.write(
    "Predict the probability that the home team wins "
    "using a Logistic Regression model with ELO ratings."
)

score_diff = st.slider(
    "Score Differential (Home - Away)",
    min_value=-20,
    max_value=20,
    value=0
)

time_remaining = st.slider(
    "Time Remaining (seconds)",
    min_value=0,
    max_value=2880,
    value=300
)

elo_diff = st.slider(
    "Pregame ELO Difference",
    min_value=-300,
    max_value=300,
    value=0
)

interaction = score_diff * time_remaining

probability = model.predict_proba(
    [[
        score_diff,
        time_remaining,
        interaction,
        elo_diff
    ]]
)[0][1]

st.metric(
    label="Home Win Probability",
    value=f"{probability:.1%}"
)