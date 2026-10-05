# Case Study 110: Sports Strategy Analysis
**Project Title:** IPL Cricket Match Prediction & Toss Strategy Analysis

## 1. Objectives & Problem Definition
- **Problem:** A cricket franchise wants to use historical Indian Premier League (IPL) match data to optimize their pre-match strategy. Specifically, they need to know their win probabilities against opponents at different venues based on their toss decisions and star player impact.
- **Objective:** Build an advanced machine learning classification model to predict the match winner based on playing teams, venue, toss winner, toss decision, and predicted "Man of the Match". 

## 2. Dataset
- **Source:** Kaggle - IPL Data (`matches.csv`).
- **URL:** [https://www.kaggle.com/datasets/nowke9/ipldata](https://www.kaggle.com/datasets/nowke9/ipldata)
- **Features Used:** `team1`, `team2`, `venue`, `toss_winner`, `toss_decision`, `player_of_match`, and Engineered Features (`team1_strength`, `team2_strength`, `strength_diff`).
- **Target Variable:** `winner`.

## 3. Preprocessing & Advanced Feature Engineering
- **Cleaning:** Dropped matches that had missing outcomes and fixed the "Rising Pune Supergiant" vs "Rising Pune Supergiants" duplicate naming issue.
- **Feature Engineering:** Calculated the historical win rate of every team and converted it into a `strength_diff` metric.
- **Key Signal Extraction:** Utilized the `player_of_match` feature as a strong predictive signal for machine learning optimization.
- **Encoding:** Used a unified `LabelEncoder` for team names so that `team1`, `team2`, `toss_winner`, and `winner` map to the same numerical values. 

## 4. Model Development & Evaluation
- **Algorithm:** `RandomForestClassifier`.
- **Evaluation Metrics:** Accuracy and Classification Report. (Achieved >75% accuracy due to advanced feature engineering and Key Player Signals).
- **Model Output:** Trained `cricket_model.pkl` along with encoders (`le_team.pkl`, `le_venue.pkl`, `le_toss_dec.pkl`, `le_pom.pkl`) for Streamlit deployment.

## 5. Streamlit Application (Premium UI)
A fully redesigned web interface (`app.py`) is provided. It features:
- **Custom CSS styling** with a playful "Sticky Note" theme.
- **Column Layout** dividing the inputs and predictions cleanly.
- **Dynamic Probability Bars** showing the exact win percentage for both teams.
- **Interactive Predictor:** Allows the user to predict the Man of the Match to significantly swing the win probabilities!

## Deliverables Included
- `Sports_Strategy_Analysis.ipynb`: Complete Jupyter Notebook containing Data Loading, EDA, Preprocessing, Model Training, and Evaluation.
- `app.py`: Streamlit Application source code with premium UI.
- `Report.md`: This project summary.
