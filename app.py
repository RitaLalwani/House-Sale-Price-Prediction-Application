import datetime
import json
import csv
import joblib
import cpi
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from joblib import dump, load
from io import StringIO
from datetime import datetime
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo
from functions import app
from functions.function import get_model_response, get_prepro_response, lgbmrpl_model


model_name = "House Sale Price Prediction Application"
model_version = "v1.0.0"
lgbmrpl_model = joblib.load('static/file/lgbmrpl_model.joblib')


app = Flask(__name__)


FIFTEEN_MINUTES = 900

@app.route('/')
@on_exception(expo, RateLimitException, max_tries=8)
@limits(calls=15, period=FIFTEEN_MINUTES)
def home():
    
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    feature_dict = request.get_json()
    
    if not feature_dict:
        return {
            'error' : 'Data not found.' 
        }, 500
        
    try: 
        response = get_model_response(feature_dict)
        
    except ValueError as e:
        return {
            'error' : str(e).split('\n')[-1].strip()
        }, 500
        
    return response, 200


@app.route('/manual', methods=['GET', 'POST'])
def manual():

    return render_template('saleprice.html')


@app.route('/saleprice', methods=['GET', 'POST'])
def saleprice():
    if request.method == "POST":
        MSSubClass = request.form.get("MSSubClass")
        MSZoning = request.form.get("MSZoning")
        Utilities = request.form.get("Utilities")
        Neighborhood = request.form.get("Neighborhood")
        BldgType = request.form.get("BldgType")
        RoofMatl = request.form.get("RoofMatl")
        Foundation = request.form.get("Foundation")
        TotalBsmtSF = request.form.get("TotalBsmtSF")
        Heating = request.form.get("Heating")
        CentralAir = request.form.get("CentralAir")
        Electrical = request.form.get("Electrical")
        FstFlrSF = request.form.get("1stFlrSF")
        LowQualFinSF = request.form.get("LowQualFinSF")
        GrLivArea = request.form.get("GrLivArea")
        FullBath = request.form.get("FullBath")
        HalfBath = request.form.get("HalfBath")
        BedroomAbvGr = request.form.get("BedroomAbvGr")
        KitchenAbvGr = request.form.get("KitchenAbvGr")
        LivRmsAbvGrd = request.form.get("LivRmsAbvGrd")
        Functional = request.form.get("Functional")
        GarageType = request.form.get("GarageType")
        PavedDrive = request.form.get("PavedDrive")
        Fence = request.form.get("Fence")
        MiscFeature = request.form.get("MiscFeature")
        MiscVal = request.form.get("MiscVal")
        SaleType = request.form.get("SaleType")
        SaleCondition = request.form.get("SaleCondition")
        StreetAlley = request.form.get("StreetAlley")
        LandSlpContr = request.form.get("LandSlpContr")
        LotCondition = request.form.get("LotCondition")
        BsmtFinType = request.form.get("BsmtFinType")
        BsmtGrade = request.form.get("BsmtGrade")
        LotExterior = request.form.get("LotExterior")
        ExterGrade = request.form.get("ExterGrade")
        GarageGrade = request.form.get("GarageGrade")
        HouseAge = request.form.get("HouseAge")
        GarageAge = request.form.get("GarageAge")
        RemodAge = request.form.get("RemodAge")
        BsmtFinSF = request.form.get("BsmtFinSF")
        TotalBsmtBath = request.form.get("TotalBsmtBath")
        WDOpnPorch = request.form.get("WDOpnPorch")
        TSSnEncPorch = request.form.get("TSSnEncPorch")
        OverallGrade = request.form.get("OverallGrade")
        X = pd.DataFrame([[MSSubClass, MSZoning, Utilities, Neighborhood, BldgType, RoofMatl, 
                Foundation, TotalBsmtSF, Heating, CentralAir, Electrical, FstFlrSF, LowQualFinSF, 
                GrLivArea, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, LivRmsAbvGrd, Functional, 
                GarageType, PavedDrive, Fence, MiscFeature, MiscVal, SaleType, SaleCondition, 
                StreetAlley, LandSlpContr, LotCondition, BsmtFinType, BsmtGrade, LotExterior, 
                ExterGrade, GarageGrade, HouseAge, GarageAge, RemodAge, BsmtFinSF, TotalBsmtBath, 
                WDOpnPorch, TSSnEncPorch, OverallGrade]], 
                columns = ["MSSubClass", "MSZoning", "Utilities", "Neighborhood", "BldgType", "RoofMatl", 
                            "Foundation", "TotalBsmtSF", "Heating", "CentralAir", "Electrical", "1stFlrSF", 
                            "LowQualFinSF", "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", 
                            "LivRmsAbvGrd", "Functional", "GarageType", "PavedDrive", "Fence", "MiscFeature", 
                            "MiscVal", "SaleType", "SaleCondition", "StreetAlley", "LandSlpContr", "LotCondition",
                            "BsmtFinType", "BsmtGrade", "LotExterior", "ExterGrade", "GarageGrade", "HouseAge",
                            "GarageAge", "RemodAge", "BsmtFinSF", "TotalBsmtBath", "WDOpnPorch", "TSSnEncPorch", 
                            "OverallGrade"])
        
        prediction = lgbmrpl_model.predict(X)[0]
        
        cpi_year = datetime.now().year - 1
        cpi_prediction = cpi.inflate(prediction, 2011, items="Housing", area="U.S. city average", to=cpi_year)

    else:
        print("Please enter valid data")

    return render_template('output.html', output='{}'.format(prediction), output1='{}'.format(cpi_prediction))


@app.route('/output', methods=['GET', 'POST'])
def output():

    return render_template('output.html')


@app.route('/csvresult', methods=['GET', 'POST'])
def csvresult():
    
    try :
        if request.method == "POST":
            rawFile = request.files['file']
            df = pd.read_csv(StringIO(rawFile.read().decode('utf-8')))
            df.to_csv("static/file/test.csv", index=False)
            df1 = pd.read_csv("static/file/test.csv")
            data = get_prepro_response(df1)
            data.to_csv('static/file/prediction.csv', float_format='%.4f', index=None)
        
            cpi_year = datetime.now().year - 1
            cpi_prediction = cpi.inflate(data, 2011, items="Housing", area="U.S. city average", to=cpi_year)
            cpi_prediction.to_csv('static/file/cpihpplgbmrcvplm.csv', float_format='%.4f', index=None)
    
    except ValueError:
            return { 'error' : 'Data file extension should be in csv format.' }, 500
    
    except:
            return { 'error': 'Mismatched Columns.'}, 200

    return render_template('csvresult.html', result=data, result1=cpi_prediction)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)