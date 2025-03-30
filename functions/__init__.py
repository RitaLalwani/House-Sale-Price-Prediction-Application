# Import libraries
import joblib
from joblib import load
from flask import Flask


# Initialize app
app = Flask(__name__,
            static_url_path='',
            static_folder='C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/static',
            template_folder='C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/templates',
            )


# Load LGBMR Model
lgbmrpl_model = joblib.load('static/file/lgbmrpl_model.joblib')
