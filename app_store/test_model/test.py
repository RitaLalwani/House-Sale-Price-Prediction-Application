import numpy as np
import pandas as pd
import joblib
import datetime
import cpi
from datetime import datetime
from joblib import dump, load


model_name = "House Sale Price Prediction Application"
model_version = "v1.0.0"
model = joblib.load('C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/app_store/app_models/lgbmrpl_model.joblib')


# Test regression model on sample data
X = pd.DataFrame([[20,2,0,12,0,0,1,882,0,1,3,896,0,896,1,0,2,1,2,6,1,2,2,0,0,8,4,2,0,2,4,3,2,4,1,49,49,49,612,0,140,120,6]], 
                columns = ["MSSubClass", "MSZoning", "Utilities", "Neighborhood", "BldgType", "RoofMatl", 
                            "Foundation", "TotalBsmtSF", "Heating", "CentralAir", "Electrical", "1stFlrSF", 
                            "LowQualFinSF", "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", 
                            "LivRmsAbvGrd", "Functional", "GarageType", "PavedDrive", "Fence", "MiscFeature", 
                            "MiscVal", "SaleType", "SaleCondition", "StreetAlley", "LandSlpContr", "LotCondition",
                            "BsmtFinType", "BsmtGrade", "LotExterior", "ExterGrade", "GarageGrade", "HouseAge",
                            "GarageAge", "RemodAge", "BsmtFinSF", "TotalBsmtBath", "WDOpnPorch", "TSSnEncPorch", 
                            "OverallGrade"])


# Light Gradient Boosting Machine Regression Model Prediction
prediction = model.predict(X)[0]


# Inflated Prediction
cpi_year = datetime.now().year - 1
cpi_prediction = cpi.inflate(prediction, 2011, items="Housing", area="U.S. city average", to=cpi_year)


# Model Info & Prediction
print("Model Name :-", model_name)
print("Model Version :-", model_version)
print("House Sale Price For Sample Data is $:-", prediction)
print("Inflated House Sale Price For Sample Data is $:-", cpi_prediction)
