# Import python libraries
import joblib
import pandas as pd
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer


# Load model
lgbmrpl_model = joblib.load('static/file/lgbmrpl_model.joblib')


# Preprocess & prediction for csv file
def get_prepro_response(data1):
    test_dummy_df = data1.copy()
    X = test_dummy_df.drop(['Id'], axis=1)

    numerical_cols = data1.select_dtypes(include="number").keys()
    categorical_cols = data1.select_dtypes(include="object").keys()
    
    numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', QuantileTransformer())
    ])

    categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="constant", fill_value='missing') ),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
    ('test_num', numerical_pipeline, numerical_cols),
    ('test_cat', categorical_pipeline, categorical_cols)
    ])
    
    X_test = X
    prediction = lgbmrpl_model.predict(X_test)
    
    prediction = pd.DataFrame({
                                "SalePrice": prediction
                            })

    return prediction


# Model functions for prediction
def predict(X, lgbmrpl_model):
    prediction = lgbmrpl_model.predict(X)
    return prediction

def get_model_response(json_data):
    X = pd.DataFrame.from_dict(json_data)
    prediction = predict(X, lgbmrpl_model)
    return prediction
