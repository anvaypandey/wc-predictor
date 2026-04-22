import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="Match Predictor",
    page_icon="🎯",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/1_Match_Predictor.py",  title="Match Predictor",   icon="🎯"),
    st.Page("pages/2_Bracket_Simulator.py", title="Bracket Simulator", icon="🏆"),
    st.Page("pages/3_Accuracy.py",          title="Model Accuracy",     icon="📊"),
])
pg.run()
