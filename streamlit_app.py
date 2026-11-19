import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('predictor.pkl')

def predict_quake(magnitude, depth, cdi, mmi, sig):
    # Prepare the feature array
    arr = np.array([magnitude, depth, cdi, mmi, sig]).reshape(1, -1)
    # Predict
    pred = model.predict(arr)
    return pred[0]

st.set_page_config(page_title="Quake Alert Predictor", page_icon="🌋")
st.title("🌋 Quake Alert Predictor")
st.write("Enter earthquake data to assess the alert level.")

# Inputs
col1, col2 = st.columns(2)
with col1:
    magnitude = st.number_input('Magnitude', min_value=0.0, value=5.0)
    depth = st.number_input('Depth', min_value=0.0, value=10.0)
    cdi = st.number_input('CDI', min_value=0.0, value=1.0)
with col2:
    mmi = st.number_input('MMI', min_value=0.0, value=1.0)
    sig = st.number_input('SIG', value=0.0)

if st.button("Predict Alert Level", type="primary"):
    result = predict_quake(magnitude, depth, cdi, mmi, sig)
    
    # Visual feedback based on result
    if result == 'green':
        st.success(f"Alert Level: {result.upper()}")
    elif result == 'yellow':
        st.warning(f"Alert Level: {result.upper()}")
    elif result == 'orange':
        st.error(f"Alert Level: {result.upper()}")
    else: # red
        st.error(f"Alert Level: {result.upper()}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>Quake Predictor App</p>
</div>
""", unsafe_allow_html=True)

