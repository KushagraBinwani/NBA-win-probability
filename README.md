# NBA Win Probability Model

## Project Overview

This project uses NBA play-by-play data to estimate the probability that the home team wins at any point during a game.

The model uses game state information such as score differential and time remaining to predict the final outcome of the game.

## Dataset

* NBA 2024 Play-by-Play Data
* 639,721 events
* 1,318 games

## Feature Engineering

Current features:

* Score Differential
* Game Seconds Remaining
* Interaction Feature (Score Differential × Time Remaining)

Planned features:

* Team ELO Ratings
* Possession
* Player Availability
* Injury Information

## Models Tested

| Model               | Features          | Accuracy |
| ------------------- | ----------------- | -------- |
| Logistic Regression | Score Diff + Time | 76.13%   |
| Logistic Regression | + Interaction     | 76.61%   |
| Random Forest       | TBD               | TBD      |

## Key Findings

* Score differential is highly predictive of game outcomes.
* Time remaining significantly affects the value of a lead.
* Interaction features improve probability calibration.

## Future Work

* Build team ELO ratings
* Add possession tracking
* Compare advanced machine learning models
* Create a live win probability dashboard
