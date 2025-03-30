# Web Application
House Sale Price Prediction Application

# Dataset
Kaggle House Price Prediction dataset

# Attribute Information
1. Id
2. MSSubClass: Identifies the type of dwelling involved in the sale.
3. MSZoning: Identifies the general zoning classification of the sale.
--------------------------------------
79. SaleType: Type of sale
80. SaleCondition: Condition of sale

# Virtual Environment
virtualenv venv
&
venv/Scripts/activate

# Python packages
pip install -r requirements.txt

# Preprocess 
app_store/train_model/preprocess.py

# Train
app_store/train_model/train.py

# Test
app_store/test_model/test.py

# Function
functions/__init__.py
&
functions/function.py

# Web Application
flask run -p 8000
&
http://127.0.0.1:8000/
&
http://127.0.0.1:8000/saleprice
& 
http://127.0.0.1:8000/output
&
http://127.0.0.1:8000/csvresult

# API Throttling
@limits(calls=15, period=FIFTEEN_MINUTES)

# Docker File Build
docker build -t hspprm .

# Docker File Test
docker run -p 8000:8000 e502a64fcad0

