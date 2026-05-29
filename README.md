# Exoplanet Hunter

This web app uses machine learning to detect exoplanets based on real data from NASA’s Kepler telescope. By analyzing measurements of stars, it predicts whether an exoplanet is present using a trained Random Forest classifier. The app also leverages Mistral AI to provide a clear, easy-to-understand scientific explanation of the prediction.

---

## Preview

![Input Form](screenshot1.png)
![Results](screenshot2.png)

---

## What it does

- Trains a Random Forest classifier using data from over 7,300 stars collected by NASA’s Kepler mission
- Reaches 89% accuracy when tested on new, unseen stars
- Shows a feature importance chart that highlights which star measurements had the biggest impact on predictions
- Uses Mistral AI to provide clear, easy-to-understand explanations of the model’s results

---

## Technologies used

| Category | Tools |
| Machine Learning | scikit-learn (Random Forest) |
| Data | pandas, numpy |
| Visualization | plotly |
| Web App | Streamlit |
| AI Explanation | Mistral AI API |
| Environment | python-dotenv |

---

## Data source

Real observational data from NASA's Kepler Space Telescope via the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu). The dataset contains 9,564 Kepler Objects of Interest (KOIs) labeled as CONFIRMED, FALSE POSITIVE, or CANDIDATE.

---

## How to run locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/exoplanet-hunter.git
cd exoplanet-hunter
```

**2. Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Mistral API key**
Create a `.env` file in the root folder:
```
MISTRAL_API_KEY=your_key_here
```
Get a free key at console.mistral.ai

**5. Download the NASA dataset**
```bash
python data_download.py
```

**6. Train the model**
```bash
python train.py
```

**7. Run the app**
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

## Project structure

```
exoplanet-hunter/
├── data/                  NASA Kepler dataset
├── model/                 saved trained model and features
├── data_download.py       downloads NASA data
├── train.py               trains and evaluates the ML model
├── explainer.py           Mistral AI explanation
├── app.py                 Streamlit web app
├── requirements.txt
└── .env                   (never shared)
```

---

## What I learned

- The basics of the Transit Method used to detect exoplanets
- How to build a full machine learning pipeline from data cleaning all the way to deployment
- How a Random Forest model identifies patterns in labeled data
- Best practices for securely calling production AI APIs
- Creating and deploying interactive web apps using Streamlit

---

*Built with real NASA Kepler Space Telescope data*