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
    label="🏀 Home Win Probability",
    value=f"{probability:.1%}"
)

st.progress(float(probability))

st.subheader("Current Game State")

st.write(f"**Score Differential:** {score_diff}")
st.write(f"**Time Remaining:** {time_remaining} seconds")
st.write(f"**ELO Difference:** {elo_diff}")

times = list(range(2880, -1, -60))

score_range = range(-20, 21)

curve = []

import matplotlib.pyplot as plt

for score in score_range:

    interaction = score * time_remaining

    p = model.predict_proba(
        [[
            score,
            time_remaining,
            interaction,
            elo_diff
        ]]
    )[0][1]

    curve.append(p)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(score_range, curve, linewidth=3)

ax.set_xlabel("Score Differential")
ax.set_ylabel("Home Win Probability")
ax.set_title("Win Probability vs Score Differential")

st.pyplot(fig)