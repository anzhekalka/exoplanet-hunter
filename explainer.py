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
    prompt = f"you are an astrophysicist explaining to a student. The developped AI model predicted that the star with the following features: {features} is a {prediction_label} with the confidence of {confidence}. explain in 3 bullet points in English why the model thinks this star has or doesn't have an exoplanet."
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

if __name__ == "__main__":
    test = explain_prediction(
        features={"koi_period": 10.5, "koi_prad": 2.3,
                  "koi_teq": 800.0, "koi_insol": 93.0,
                  "koi_model_snr": 45.0, "koi_steff": 5455.0,
                  "koi_slogg": 4.467, "koi_srad": 0.927},
        prediction=1,
        confidence=0.87
    )
    print(test)