# Case Study 110: Sports Strategy Analysis
**Project Title:** IPL Cricket Match Prediction & Toss Strategy Analysis

## 1. Objectives & Problem Definition
- **Problem:** A cricket franchise wants to use historical Indian Premier League (IPL) match data to optimize their pre-match strategy. Specifically, they need to know their win probabilities against opponents at different venues based on their toss decisions.
- **Objective:** Build an advanced machine learning classification model to predict the match winner using purely pre-match data to establish a realistic baseline accuracy for sports prediction.

## 2. Dataset
- **Source:** Streamlined IPL Dataset (`simple_matches.csv`).
- **Features Used:** `team1`, `team2`, `venue`, `toss_winner`, `toss_decision`.
- **Target Variable:** `winner`.

## 3. Preprocessing
- **Cleaning:** Dropped matches that had missing outcomes and ensured consistent team naming conventions.
- **Encoding:** Used a unified `LabelEncoder` for team names so that `team1`, `team2`, `toss_winner`, and `winner` map to the same numerical entities. 

## 4. Model Development & Meaningful Experiments
To fulfill the objective of comparing meaningful experiments, we evaluated multiple models on the pre-match data:

### Experiment: Authentic Pre-Match Prediction
- **Goal:** Predict the winner using strictly pre-match data (Teams, Venue, Toss). 
- **Models Compared:** `LogisticRegression` vs `RandomForestClassifier`.
- **Result:** Both models achieved a realistic ~50-60% accuracy. This authentically reflects the highly volatile, unpredictable nature of T20 cricket where pre-match features alone cannot definitively predict outcomes.

## 5. Model Selection & Deployment
We selected the **Random Forest Classifier** as our primary predictor due to its ability to capture non-linear interactions between venue conditions and team pairings.
- **Deployed Files:** `cricket_model.pkl`, `le_team.pkl`, `le_venue.pkl`, `le_toss_dec.pkl`.

## 6. Streamlit Application (Premium UI)
A fully redesigned web interface (`app.py`) is provided for the Authentic Pre-Match Predictor. It features:
- **Custom CSS styling** with a playful "Sticky Note" theme.
- **Dynamic Probability Breakdown** showing realistic win percentages based purely on pre-match strategic indicators.

## Deliverables Included
- `simple_matches.csv`: Simplified, clean dataset ready for presentation.
- `Sports_Strategy_Analysis.ipynb`: Complete Jupyter Notebook containing Data Loading, Preprocessing, Experimental Comparisons, and Evaluation.
- `app.py`: Streamlit Application source code.
- `Report.md`: This project summary.
