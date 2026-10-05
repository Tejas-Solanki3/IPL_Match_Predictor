import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------- UI Config -----------------
st.set_page_config(page_title="IPL Match Predictor", layout="wide")

# Custom CSS for Sticky Note Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600&family=Poppins:wght@400;600&display=swap');

    .title-wrapper {
        text-align: center;
        margin-bottom: 40px;
    }
    
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 50px;
        font-weight: 800;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sticky Note Styling */
    .sticky-note {
        padding: 30px;
        margin: 15px;
        border-radius: 2px 15px 15px 15px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.15);
        font-family: 'Caveat', cursive;
        font-size: 24px;
        color: #333 !important;
        position: relative;
        transition: transform 0.2s;
    }
    .sticky-note:hover {
        transform: scale(1.02);
    }
    
    .sticky-yellow { background: #fdfd96; transform: rotate(-1deg); }
    .sticky-pink   { background: #ffb7b2; transform: rotate(1deg); }
    .sticky-blue   { background: #b5ead7; transform: rotate(-2deg); }
    
    .sticky-note h4 {
        font-family: 'Poppins', sans-serif;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 10px;
        color: #222 !important;
        text-transform: uppercase;
    }
    
    /* Result Card */
    .prediction-board {
        background-color: #2c3e50;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-top: 20px;
        border: 4px dashed #fdfd96;
    }
    
    .prediction-text {
        font-family: 'Poppins', sans-serif;
        font-size: 42px;
        font-weight: 800;
        color: #fdfd96;
        margin: 15px 0;
    }

    /* Clean up native inputs */
    .stSelectbox label, .stRadio label {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    
    .stButton>button {
        background-color: #ff8a00;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
        width: 100%;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e52e71;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load('cricket_model.pkl')
        le_team = joblib.load('le_team.pkl')
        le_venue = joblib.load('le_venue.pkl')
        le_toss_dec = joblib.load('le_toss_dec.pkl')
        return model, le_team, le_venue, le_toss_dec
    except Exception as e:
        return None, None, None, None

model, le_team, le_venue, le_toss_dec = load_models()

# ----------------- App Layout -----------------
st.markdown("<div class='title-wrapper'><span class='main-title'>IPL Match Strategy Board</span></div>", unsafe_allow_html=True)

if model is None:
    st.error("Models not found. Please run the Jupyter Notebook first to train the authentic pre-match model.")
    st.stop()

# Info row with Sticky Notes
note1, note2, note3 = st.columns(3)

with note1:
    st.markdown("""
    <div class="sticky-note sticky-yellow">
        <h4>What we Predict</h4>
        We predict the absolute winner of an IPL match based strictly on pre-match indicators. Will the home team dominate, or will the visitors steal the show?
    </div>
    """, unsafe_allow_html=True)

with note2:
    st.markdown("""
    <div class="sticky-note sticky-pink">
        <h4>Our Features</h4>
        - Team 1 & Team 2<br>
        - Match Venue<br>
        - Toss Winner<br>
        - Toss Decision
    </div>
    """, unsafe_allow_html=True)

with note3:
    st.markdown("""
    <div class="sticky-note sticky-blue">
        <h4>The Strategy</h4>
        T20 Cricket is highly unpredictable! We utilize an Authentic Pre-Match Model (Random Forest) that avoids data leakage to provide realistic win probabilities based purely on the toss and venue.
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Input layout
col_input, col_result = st.columns([1, 1.2])

with col_input:
    st.markdown("### Match Setup")
    team_options = le_team.classes_
    venue_options = le_venue.classes_
    toss_dec_options = le_toss_dec.classes_
    
    team1 = st.selectbox("Select Team 1", team_options)
    team2 = st.selectbox("Select Team 2", team_options, index=1 if len(team_options)>1 else 0)
    
    venue = st.selectbox("Select Match Venue", venue_options)
    
    st.markdown("### Toss Strategy")
    toss_winner = st.radio("Who won the toss?", (team1, team2), horizontal=True)
    toss_decision = st.radio("Toss Decision", toss_dec_options, horizontal=True)
    
    st.write("")
    predict_btn = st.button("Analyze Match")

with col_result:
    if predict_btn:
        if team1 == team2:
            st.error("Teams must be different!")
        else:
            with st.spinner('Running Authentic Pre-Match Analysis...'):
                # Encode base inputs
                t1_enc = le_team.transform([team1])[0]
                t2_enc = le_team.transform([team2])[0]
                v_enc = le_venue.transform([venue])[0]
                tw_enc = le_team.transform([toss_winner])[0]
                td_enc = le_toss_dec.transform([toss_decision])[0]

                # Formulate input exactly as trained (Experiment A)
                input_data = pd.DataFrame({
                    'team1_enc': [t1_enc],
                    'team2_enc': [t2_enc],
                    'venue_enc': [v_enc],
                    'toss_winner_enc': [tw_enc],
                    'toss_decision_enc': [td_enc]
                })
                
                prediction = model.predict(input_data)
                predicted_class = le_team.inverse_transform(prediction)[0]
                
                probabilities = model.predict_proba(input_data)[0]
                prob_dict = {cls: prob for cls, prob in zip(le_team.classes_, probabilities) if cls in [team1, team2]}
                
                total = sum(prob_dict.values())
                if total > 0:
                    win_prob1 = (prob_dict[team1] / total) * 100
                    win_prob2 = (prob_dict[team2] / total) * 100
                else:
                    win_prob1 = 50.0
                    win_prob2 = 50.0

            st.markdown(f"""
            <div class="prediction-board">
                <p style="font-size: 20px; text-transform: uppercase; letter-spacing: 3px; margin:0;">AI Predicted Winner</p>
                <div class="prediction-text">{predicted_class}</div>
                <hr style="border-color: #555;">
                <p style="font-size: 18px;">Win Probability Breakdown</p>
                <p style="font-size: 22px; font-weight: bold; margin: 5px 0;">{team1}: <span style="color:#fdfd96;">{win_prob1:.1f}%</span></p>
                <p style="font-size: 22px; font-weight: bold; margin: 5px 0;">{team2}: <span style="color:#ffb7b2;">{win_prob2:.1f}%</span></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height: 100%; display: flex; align-items: center; justify-content: center; border: 2px dashed #888; border-radius: 15px; padding: 50px; margin-top: 30px;">
            <h3 style="color: #888; text-align: center;">Set up the match on the left to see the prediction board.</h3>
        </div>
        """, unsafe_allow_html=True)
