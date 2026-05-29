"""
explainer.py sends the request to teh Mistral AI with the star measurements and the model answers.
it fetches the AI's answer 
"""

import requests
import os
from dotenv import load_dotenv 

load_dotenv() #load .env in the memory

api_key = os.getenv("MISTRAL_API_KEY") #stores API in a variable


def explain_prediction(features: dict,  prediction: int ,confidence: float) : 
    """
    returns the explanaition provided by the Mistral AI. Returns None if the request was not successful 
    """
    prediction_label = "CONFIRMED exoplanet" if prediction == 1 else "FALSE POSITIVE" 
    prompt = f"""You are an astrophysicist explaining to a student.
    The AI model predicted this star is a {prediction_label} with confidence {confidence}.

    Star measurements:
    - Orbital Period: {features.get('koi_period')} days
    - Planetary Radius: {features.get('koi_prad')} Earth radii
    - Equilibrium Temperature: {features.get('koi_teq')} K
    - Insolation Flux: {features.get('koi_insol')} Earth flux
    - Signal-to-Noise Ratio: {features.get('koi_model_snr')}
    - Stellar Temperature: {features.get('koi_steff')} K
    - Stellar Surface Gravity: {features.get('koi_slogg')}
    - Stellar Radius: {features.get('koi_srad')} Solar radii

    Explain in 3 bullet points why the model thinks this star has or doesn't have an exoplanet.
    Use the readable names above, not the code names like koi_prad."""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    } #headers for the requests.post()
    body = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400
    } #json for the requests.post()
    url = 'https://api.mistral.ai/v1/chat/completions'
    responce = requests.post(url, headers = headers, json = body) #sending the request to Mistral AI
    if responce.status_code == 200: 
        final_message = responce.json()["choices"][0]["message"]["content"] #json() conversion + extracting the content from a dictionary
        return final_message
    else:
        print ("API error:", responce.status_code, responce.text)
        return None 
