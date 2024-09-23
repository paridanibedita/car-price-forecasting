##BUSINESS PROBLEM:
'''The price of a car depends on a lot of factors like the goodwill of the brand of the car,
features of the car, horsepower and the mileage it gives and many more.prediction of car price by using ML models.
'''
##BUSINESS OBJECTIVE:
'''Predict the car price by considering some features.'''
##BUSINESS CONSTRAINTS:
'''increase the customer satisfaction.'''

##BUSINESS SUCCESS CRITERIA:
'''
1.Business success criteria-The model helps sellers and dealers optimize pricing, leading to increased profits.
2.Economic success criteria-: By providing accurate pricing, it encourages more transactions between buyers and sellers.
3.ML success criteria-Achieve an accuray of at least 95%.'''

##Data dictionary
'''Columns:
Car_Name: Name of the car (categorical)
Year: Year of the car's manufacturing (numerical)
Selling_Price: Selling price of the car (target variable, numerical)
Present_Price: Current market price of the car (numerical)
Driven_kms: Kilometers driven by the car (numerical)
Fuel_Type: Type of fuel used by the car (categorical)
Selling_type: Selling type (categorical)
Transmission: Type of transmission (categorical)
Owner: Number of previous owners (numerical)'''
    
##import the necessary libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from feature_engine.outliers import Winsorizer
import joblib

##load dataset
data = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Documents\Data science\project2\TASK3\car data.csv")
data

##description and information about the data
data.describe()
data.info()

##check how many number of cars are there in each car segment
data['Car_Name'].value_counts()

##check is there any null values
data.isnull().sum()

##create histogram to check outliers and skewness of each numeric columns
columns = ['Selling_Price',"Present_Price",'Driven_kms','Owner']

for column in columns:
    plt.hist(data[column], bins=30)  # Adjust the number of bins as needed
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'{column} - Histogram')
    plt.show()
    
##auto EDA
#sweetviz
import sweetviz
my_report = sweetviz.analyze([data,"data"])
my_report.show_html("report.html")

#D-tale
import dtale
d = dtale.show(data)
d.open_browser()
    
##Data Cleaning and Preprocessing

##using boxplot to determine outlietrs
data.plot(kind = "box",subplots = True,sharey = False,figsize =(15,13))
plt.subplots_adjust(wspace = 1.5) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()
   

# Outlier treatment using Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Present_Price','Selling_Price', 'Driven_kms'])
data[['Present_Price','Selling_Price', 'Driven_kms']] = winsor.fit_transform(data[['Present_Price','Selling_Price', 'Driven_kms']])

# Drop 'Car_Name' (not needed for prediction)
X = data.drop(['Selling_Price', 'Car_Name'], axis=1)
y = data['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines for numeric and categorical data
numeric_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner']
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)])

##Stacking Ensemble Model 
# Base models for stacking
base_models = [
    ('decision_tree', DecisionTreeRegressor(random_state=42)),
    ('knn', KNeighborsRegressor()),
    ('random_forest', RandomForestRegressor(random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(random_state=42)),
    ('xgboost', XGBRegressor(random_state=42))
]

# Meta-model (final model)
meta_model = LinearRegression()

# Stacking Regressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Build a pipeline for stacking
stacking_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('stacking', stacking_model)])

# Train the stacking ensemble model
stacking_pipeline.fit(X_train, y_train)

# Evaluate the performance of the model
y_pred = stacking_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE) for Stacking Model: {rmse}")

## Save the Model

# Save the preprocessing pipeline and stacking model to a file
joblib.dump(stacking_pipeline, 'stacking_pipeline.pkl')    























































































