# 1. Import Python Libraries
import math
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import OrdinalEncoder

# 2. Load Dataset
train_df = pd.read_csv("C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/dataset/train.csv")
test_df = pd.read_csv("C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/dataset/test.csv")

# 3. Fill Missing Values
# Fill train dataset num & cat column's values with median number & missing word
train_df[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']] = train_df[['LotFrontage', 'MasVnrArea', 
            'GarageYrBlt']].fillna(train_df[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].median())

train_df = train_df.fillna("missing")

# Fill test dataset num & cat column's values with median number & missing word
test_df[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
            'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']] = test_df[['LotFrontage', 'MasVnrArea', 
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 
            'GarageCars', 'GarageArea']].fillna(test_df[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
            'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']].median())

test_df = test_df.fillna("missing")


# 4. Feature Engineering
# Use OrdinalEncoder to convert categorical columns categories into numerical values for train/test dataset
# Initialize OrdinalEncoder
train_oe = OrdinalEncoder()
test_oe = OrdinalEncoder()

# Fit the encoder on the categorical columns
train_df[['Street', 'Alley', 'LandSlope', 'LandContour', 'Condition1', 'Condition2', 
        'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'Exterior1st', 'Exterior2nd', 
        'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond']] = train_oe.fit_transform(train_df[['Street', 
        'Alley', 'LandSlope', 'LandContour', 'Condition1', 'Condition2', 
        'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'Exterior1st', 'Exterior2nd', 
        'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond']])
        
test_df[['Street', 'Alley', 'LandSlope', 'LandContour', 'Condition1', 'Condition2', 
        'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'Exterior1st', 'Exterior2nd', 
        'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond']] = test_oe.fit_transform(test_df[['Street', 
        'Alley', 'LandSlope', 'LandContour', 'Condition1', 'Condition2', 'BsmtFinType1', 'BsmtFinType2', 
        'BsmtQual', 'BsmtCond', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond']])


# Merge Categorical columns in train/test dataset
train_df['StreetAlley'] = 0.5 * (train_df['Street']) + 0.5 * (train_df['Alley'])
train_df['LandSlpContr'] = 0.5 * (train_df['LandSlope']) + 0.5 * (train_df['LandContour'])
train_df['LotCondition'] = 0.5 * (train_df['Condition1']) + 0.5 * (train_df['Condition2'])
train_df['BsmtFinType'] = 0.5 * (train_df['BsmtFinType1']) + 0.5 * (train_df['BsmtFinType2'])
train_df['BsmtGrade'] = 0.5 * (train_df['BsmtQual']) + 0.5 * (train_df['BsmtCond'])
train_df['LotExterior'] = 0.5 * (train_df['Exterior1st']) + 0.5 * (train_df['Exterior2nd'])
train_df['ExterGrade'] = 0.5 * (train_df['ExterQual']) + 0.5 * (train_df['ExterCond'])
train_df['GarageGrade'] = 0.5 * (train_df['GarageQual']) + 0.5 * (train_df['GarageCond'])

test_df['StreetAlley'] = 0.5 * (test_df['Street']) + 0.5 * (test_df['Alley'])
test_df['LandSlpContr'] = 0.5 * (test_df['LandSlope']) + 0.5 * (test_df['LandContour'])
test_df['LotCondition'] = 0.5 * (test_df['Condition1']) + 0.5 * (test_df['Condition2'])
test_df['BsmtFinType'] = 0.5 * (test_df['BsmtFinType1']) + 0.5 * (test_df['BsmtFinType2'])
test_df['BsmtGrade'] = 0.5 * (test_df['BsmtQual']) + 0.5 * (test_df['BsmtCond'])
test_df['LotExterior'] = 0.5 * (test_df['Exterior1st']) + 0.5 * (test_df['Exterior2nd'])
test_df['ExterGrade'] = 0.5 * (test_df['ExterQual']) + 0.5 * (test_df['ExterCond'])
test_df['GarageGrade'] = 0.5 * (test_df['GarageQual']) + 0.5 * (test_df['GarageCond'])

# Merge Numerical columns in train/test dataset
train_df['HouseAge'] = train_df['YrSold'] - train_df['YearBuilt']
train_df['GarageAge'] = train_df['YrSold'] - train_df['GarageYrBlt']
train_df['RemodAge'] = train_df['YrSold'] - train_df['YearRemodAdd']
train_df['BsmtFinSF'] = train_df['BsmtFinSF1'] + train_df['BsmtFinSF2']
train_df['TotalBsmtBath'] = train_df['BsmtFullBath'] + 0.5 * (train_df['BsmtHalfBath'])
train_df['LivRmsAbvGrd'] = train_df['TotRmsAbvGrd'] - (train_df['BedroomAbvGr'] + train_df['KitchenAbvGr'])
train_df['WDOpnPorch'] = train_df['OpenPorchSF'] + train_df['WoodDeckSF']
train_df['TSSnEncPorch'] = train_df['EnclosedPorch'] + train_df['3SsnPorch'] + train_df['ScreenPorch']
train_df['OverallGrade'] = 0.5 * (train_df['OverallQual']) + 0.5 * (train_df['OverallCond'])

test_df['HouseAge'] = test_df['YrSold'] - test_df['YearBuilt']
test_df['GarageAge'] = test_df['YrSold'] - test_df['GarageYrBlt']
test_df['RemodAge'] = test_df['YrSold'] - test_df['YearRemodAdd']
test_df['BsmtFinSF'] = test_df['BsmtFinSF1'] + test_df['BsmtFinSF2']
test_df['TotalBsmtBath'] = test_df['BsmtFullBath'] + 0.5 * (test_df['BsmtHalfBath'])
test_df['LivRmsAbvGrd'] = test_df['TotRmsAbvGrd'] - (test_df['BedroomAbvGr'] + test_df['KitchenAbvGr'])
test_df['WDOpnPorch'] = test_df['OpenPorchSF'] + test_df['WoodDeckSF']
test_df['TSSnEncPorch'] = test_df['EnclosedPorch'] + test_df['3SsnPorch'] + test_df['ScreenPorch']
test_df['OverallGrade'] = 0.5 * (test_df['OverallQual']) + 0.5 * (test_df['OverallCond'])


# Drop the extra numerical columns from train/test dataset
train_df.drop(['LotFrontage', 'LotArea', 'OverallQual', 'MasVnrArea', 'BsmtFinSF2', 
                '2ndFlrSF', 'BsmtUnfSF', 'BsmtHalfBath', 'Fireplaces', 'GarageCars', 'GarageArea', 
                'WoodDeckSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MoSold', 
                'YrSold', 'YearBuilt', 'GarageYrBlt', 'YearRemodAdd', 'BsmtFullBath', 'TotRmsAbvGrd',
                'BsmtFinSF1', 'OpenPorchSF', 'OverallCond'], axis=1, inplace=True)

test_df.drop(['LotFrontage', 'LotArea', 'OverallQual', 'MasVnrArea', 'BsmtFinSF2', 
                '2ndFlrSF', 'BsmtUnfSF', 'BsmtHalfBath', 'Fireplaces', 'GarageCars', 'GarageArea', 
                'WoodDeckSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MoSold', 
                'YrSold', 'YearBuilt', 'GarageYrBlt', 'YearRemodAdd', 'BsmtFullBath', 'TotRmsAbvGrd',
                'BsmtFinSF1', 'OpenPorchSF', 'OverallCond'], axis=1, inplace=True)

# Drop the extra categorical columns from train/test dataset
train_df.drop(['LandSlope', 'Condition1', 'BsmtFinType2','Street', 'Alley', 'LandContour', 'Condition2', 
                'BsmtFinType1', 'BsmtQual', 'BsmtCond', 'ExterQual', 'ExterCond', 'PoolQC',  
                'Exterior1st', 'Exterior2nd', 'LotShape', 'LotConfig', 'HouseStyle', 'RoofStyle', 
                'MasVnrType', 'BsmtExposure', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 
                'GarageQual', 'GarageCond',], axis=1, inplace=True)

test_df.drop(['LandSlope', 'Condition1', 'BsmtFinType2','Street', 'Alley', 'LandContour', 'Condition2', 
                'BsmtFinType1', 'BsmtQual', 'BsmtCond', 'ExterQual', 'ExterCond', 'PoolQC',  
                'Exterior1st', 'Exterior2nd', 'LotShape', 'LotConfig', 'HouseStyle', 'RoofStyle', 
                'MasVnrType', 'BsmtExposure', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 
                'GarageQual', 'GarageCond',], axis=1, inplace=True)


# Save preprocessed train/test dataset
train_df.to_csv('C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/app_store/app_dataset/train_pp_df.csv', index=False)
test_df.to_csv('C:/Users/rital/python_projects/Hackathons/HousePricePredictionApp/app_store/app_dataset/test_pp_df.csv', index=False)

# View Shape and top 5 rows of train/test dataset
print("Train/Test dataset rows and columns shape :-", train_df.shape, test_df.shape)
print("\nTrain dataset top 5 rows :-", train_df.head())
print("\nTest dataset top 5 rows :-", test_df.head())
