""" app.py - Streamlit web app for the Exoplanet Hunter project.
Loads the trained Random Forest model, takes stellar measurements as input,
predicts whether a star hosts an exoplanet, and displays a Mistral AI explanation.
"""

#imports
from explainer import explain_prediction
import streamlit as st
import plotly.graph_objects as go 
import joblib 
import pandas as pd
# streamlit run app.py

#loading the model and features 
@st.cache_resource
def load_model(): 
    model = joblib.load('model/model.pkl')
    features = joblib.load('model/features.pkl')
    return model, features

#webpage
model, features = load_model()
st.set_page_config(page_title="Exoplanet Detection Model", page_icon="🪐")
st.title("Exoplanet Detector AI")
st.write("This web app features a supervised AI model designed to detect the potential existence of an exoplanet based on a star's features. Simply input the star's data, and the model will provide a clear prediction along with a certainty score to indicate confidence in the detection. Additionally, a detailed explanation of the model's reasoning is presented, supported by a visual plot that highlights key data points influencing the prediction. Test the model yourself and explore the exciting frontier of exoplanet discovery with cutting-edge AI technology.")
st.divider()
col1, col2 = st.columns(2)
with col1: 
    st.subheader("Planet measurements")
    period = st.number_input("Orbital Period (days)", value=10.5)
    prad = st.number_input("Planetary Radius (Earth radii)", value=2.3)
    teq = st.number_input("Equilibrium Temperature (Kelvin)", value=800.0)
    insol = st.number_input("Insolation Flux (Earth flux)", value=93.0)
    model_snr = st.number_input("Transit Signal-to-Noise Ratio", value=45.0)
with col2:
    st.subheader("Star measurements")
    steff = st.number_input("Stellar Effective Temperature (Kelvin)", value=5455.0)
    slogg = st.number_input("Stellar Surface Gravity (log g)", value=4.467)
    srad = st.number_input("Stellar Radius (Solar radii)", value=0.927)

if st.button("Analyze"):
    input_data = {
        "koi_period": period, 
        "koi_prad": prad, 
        "koi_teq": teq, 
        "koi_insol": insol, 
        "koi_model_snr": model_snr, 
        "koi_steff": steff, 
        "koi_slogg": slogg, 
        "koi_srad": srad
    }

    prediction = model.predict(pd.DataFrame([input_data]))[0]
    confidence = model.predict_proba(pd.DataFrame([input_data]))[0]
    if prediction == 1 : 
        st.success("Exoplanet confirmed!")
    else: 
        st.error("False positive - no exoplanet detected")
    confidence_score = confidence[1] if prediction == 1 else confidence[0]
    with st.spinner("Aligning the telescopes..."):
        st.metric(label="Model Confidence", value=f"{confidence_score:.1%}")
        st.progress(confidence_score)
    

    
    feature_importance = model.feature_importances_
    with st.spinner("Scanning distant solar systems..."):
        fig = go.Figure(go.Bar(
            x = features, 
            y = feature_importance, 
            marker_color='#3a7bd5'
        ))
        fig.update_layout(title="What influenced this prediction?")
        st.plotly_chart(fig, use_container_width=True)

    with st.spinner("Consulting the stars..."):
        explanation = explain_prediction(input_data, prediction, confidence_score)
        if explanation == None : 
            st.warning("Our starship hit an asteroid—API request failed. Check your connection and try again.")
        else: 
            st.markdown(explanation)