# Import Python Libraries
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from numpy import sqrt, round
from PIL import Image


# Load Dataset
train_pp_df = pd.read_csv("C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/app_store/app_dataset/train_pp_df.csv")
test_pp_df = pd.read_csv("C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/app_store/app_dataset/test_pp_df.csv")


# Find the numerical & categorical columns name in train/test dataset
train_num_df = train_pp_df.select_dtypes(include="number").keys()
print("Train dataset numerical columns shape :-", train_num_df.shape)
print("\nTrain dataset numerical columns name :-", train_num_df)

print("******************************")

train_cat_df = train_pp_df.select_dtypes(include="object").keys()
print("Train dataset categorical columns shape :-", train_cat_df.shape)
print("\nTrain dataset categorical columns name :-", train_cat_df)

print("******************************")

test_num_df = test_pp_df.select_dtypes(include="number").keys()
print("Test dataset numerical columns shape :-", test_num_df.shape)
print("\nTest dataset numerical columns name :-", test_num_df)

print("******************************")

test_cat_df = test_pp_df.select_dtypes(include="object").keys()
print("Test dataset categorical columns shape :-", test_cat_df.shape)
print("\nTest dataset categorical columns name :-", test_cat_df)


# Y_test to match with X_test count
train_dummy_df = train_pp_df.copy()

X = train_dummy_df.drop(['Id', 'SalePrice'], axis=1)
Z = test_pp_df.drop('Id', axis=1)
Y = train_dummy_df['SalePrice']

values = [496]
train_dummy_df = train_dummy_df[train_dummy_df.Id.isin(values) == False]
V = train_dummy_df["SalePrice"]


# Numerical & Categorical columns name for train dataset
train_numerical_cols = ['MSSubClass', 'TotalBsmtSF', '1stFlrSF', 'LowQualFinSF',
       'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'MiscVal', 'StreetAlley', 'LandSlpContr', 'LotCondition',
       'BsmtFinType', 'BsmtGrade', 'LotExterior', 'ExterGrade', 'GarageGrade',
       'HouseAge', 'GarageAge', 'RemodAge', 'BsmtFinSF', 'TotalBsmtBath',
       'LivRmsAbvGrd', 'WDOpnPorch', 'TSSnEncPorch', 'OverallGrade']

train_categorical_cols = ['MSZoning', 'Utilities', 'Neighborhood', 'BldgType', 'RoofMatl',
       'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional',
       'GarageType', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType',
       'SaleCondition']


# Numerical & Categorical columns name for test dataset
test_numerical_cols = ['MSSubClass', 'TotalBsmtSF', '1stFlrSF', 'LowQualFinSF',
       'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'MiscVal', 'StreetAlley', 'LandSlpContr', 'LotCondition', 'BsmtFinType',
       'BsmtGrade', 'LotExterior', 'ExterGrade', 'GarageGrade', 'HouseAge',
       'GarageAge', 'RemodAge', 'BsmtFinSF', 'TotalBsmtBath', 'LivRmsAbvGrd',
       'WDOpnPorch', 'TSSnEncPorch', 'OverallGrade']

test_categorical_cols = ['MSZoning', 'Utilities', 'Neighborhood', 'BldgType', 'RoofMatl',
       'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional',
       'GarageType', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType',
       'SaleCondition']


# Numerical & Categorical pipeline building for train dataset
train_numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', QuantileTransformer())
])

train_categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="constant", fill_value='missing') ),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])


# Numerical & Categorical pipeline building for test dataset
test_numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', QuantileTransformer())
])

test_categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="constant", fill_value='missing') ),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])


# Combine Numerical & Categorical pipeline in train/test dataset
train_preprocessor = ColumnTransformer(transformers=[
    ('train_num', train_numerical_pipeline, train_numerical_cols),
    ('train_cat', train_categorical_pipeline, train_categorical_cols)
]) 

test_preprocessor = ColumnTransformer(transformers=[
    ('test_num', test_numerical_pipeline, test_numerical_cols),
    ('test_cat', test_categorical_pipeline, test_categorical_cols)
])


# Building model pipeline for train dataset
train_pipeline = Pipeline(steps=[
    ('train_preprocessor', train_preprocessor),
    ('train_model', LGBMRegressor())
])


# Train/Test/Split Dataset
X_train = X
X_test = Z
y_train = Y
y_test = V


# Train model
lgbmrpl_model = train_pipeline.fit(X_train, y_train)


# Save LGBM Regressor Pipeline Model Inside app_model Folder
model_path = 'C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/app_store/app_models/lgbmrpl_model.joblib'
dump(lgbmrpl_model, model_path)

# Save LGBM Regressor Pipeline Model Inside Static File Folder
model_path = 'C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/static/file/lgbmrpl_model.joblib'
dump(lgbmrpl_model, model_path)


# Predict model
prediction = lgbmrpl_model.predict(X_test)


# Metric Result
accuracy = round(lgbmrpl_model.score(X_test, y_test) * 100, 2)
print("The Accuracy of the trained model is:-", accuracy)

mse = mean_squared_error(y_test, prediction)
print("The MSE of the model is:-", mse)

rmse = sqrt(mean_squared_error(y_test, prediction))
print("The RMSE of the model is:-", rmse)

mae = mean_absolute_error(y_test, prediction)
print("The MAE of the model is:-", mae)

r2 = r2_score(y_test, prediction)
print("The R2 of the model is:-", r2)

model_prediction = pd.DataFrame({
                                    "Id": test_pp_df["Id"],
                                    "SalePrice": prediction
                                    })

model_prediction.to_csv('C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/app_store/app_models/kglhpplgbmrplm_submission.csv', float_format='%.4f', index=False)
print(model_prediction.head())

# Kaggle Score Screenshot of LGBMR Pipeline Model
image = Image.open('C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/app_store/app_images/kglhpplgbmrplm_submission_score.png')
image.show()
