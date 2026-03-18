import streamlit as st
import pandas as pd
import joblib

# load files
model = joblib.load("football_model.pkl")
features = joblib.load("features.pkl")
df = pd.read_csv("processed_matches.csv")

st.title("⚽ Football Match Predictor")
st.caption("Built by Junaid | ML Football Predictor ⚽")
st.warning(
    """
⚠️ Disclaimer:

This app is built for educational purposes only.

• Uses static historical data (not real-time)  
• Match results are fixed and not updated live  
• Predictions are based on a simple ML model  

🚀 With real-time data APIs, this can be upgraded to a live prediction system.

*(Until then... waiting for football federations 😄)*
"""
)

teams = sorted(df['home_team'].unique())

home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

# feature function
def get_team_features(team, df, last_n=5):
    matches = df[
        (df['home_team'] == team) |
        (df['away_team'] == team)
    ].sort_values('date', ascending=False).head(last_n)

    goals = []
    conceded = []
    points = []
    results = []  


    for _, row in matches.iterrows():
        if row['home_team'] == team:
            gf = row['home_team_goal']
            ga = row['away_team_goal']
        else:
            gf = row['away_team_goal']
            ga = row['home_team_goal']

        goals.append(gf)
        conceded.append(ga)

        if gf > ga:
            points.append(3)
            results.append("W")
        elif gf == ga:
            points.append(1)
            results.append("D")
        else:
            points.append(0)
            results.append("L")

    avg_goals = sum(goals)/len(goals) if goals else 0
    avg_conceded = sum(conceded)/len(conceded) if conceded else 0
    form_score = sum(points)
    win_rate = form_score / 15

    return form_score, avg_goals, avg_conceded, win_rate, results

# predict
if st.button("Predict Match", key="predict_button"):

    hf, hg, hc, hw, hr = get_team_features(home_team, df)
    af, ag, ac, aw, ar = get_team_features(away_team, df)

    input_data = pd.DataFrame([{
        'form_diff': hf - af,
        'goal_diff_avg': hg - ag,
        'goal_diff_form': (hg - hc) - (ag - ac),
        'win_rate_diff': hw - aw
    }])

    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]

    if prediction == 0:
        result = "🏠 Home Win"
    elif prediction == 1:
        result = "🤝 Draw"
    else:
        result = "✈️ Away Win"

    # 🔥 UI
    st.subheader("📊 Match Prediction")
    st.success(f"Prediction: {result}")

    st.subheader("📈 Confidence")

    home_prob = probs[0]
    draw_prob = probs[1]
    away_prob = probs[2]

    # HOME
    st.markdown("**🏠 Home Win**")
    st.progress(float(home_prob))
    st.markdown(f"<span style='color:green;'>{home_prob*100:.1f}%</span>", unsafe_allow_html=True)

    # DRAW
    st.markdown("**🤝 Draw**")
    st.progress(float(draw_prob))
    st.markdown(f"<span style='color:orange;'>{draw_prob*100:.1f}%</span>", unsafe_allow_html=True)

    # AWAY
    st.markdown("**✈️ Away Win**")
    st.progress(float(away_prob))
    st.markdown(f"<span style='color:red;'>{away_prob*100:.1f}%</span>", unsafe_allow_html=True)
      

    # 🔥 FORM DISPLAY (INSIDE BUTTON)
    st.subheader("📊 Recent Form (Last 5 Matches)")

    def format_results(results):
        styled = ""
        for r in results:
            if r == "W":
                styled += "<span style='color:green; font-weight:bold; font-size:18px;'>W</span> "
            elif r == "D":
                styled += "<span style='color:orange; font-weight:bold; font-size:18px;'>D</span> "
            else:
                styled += "<span style='color:red; font-weight:bold; font-size:18px;'>L</span> "
        return styled

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{home_team}**", unsafe_allow_html=True)
        st.markdown(format_results(hr), unsafe_allow_html=True)

    with col2:
        st.markdown(f"**{away_team}**", unsafe_allow_html=True)
        st.markdown(format_results(ar), unsafe_allow_html=True)