# Projects Overview
There are 3 projects under this repo:
- Predicting House Prices using Linear Regression
- Wine Quality Prediction
- Google Playstore Data Analysis

## 1 - Predicting House Prices with Linear Regression
(Project 1 Proposal. Level 2.)

### Overview
The main goal of this project is to build a predictive model using Linear Regression to estimate house prices based on certain features.

### Data Sources
Kaggle Dataset: The project utilises a publicly available housing dataset found on Kaggle.

### Tools
Python - EDA, Cleaning, Analysis and Visualisation.
- Pandas
- Seaborn
- Matplotlib
- Sci-kit Learn

### Data Preparation and Cleaning
To get started, I imported all the relevant libraries:
```
# Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from skelearn.metrics import r2_score
```
I then proceeded to load the data:
```
housing = pd.DataFrame(pd.read_csv("Housing.csv")
```

To get a glance at the data, I performed some simple commands:
- For a summary of the data, I used the describe() function in Pandas
```
housing.describe()
```
- I looked at the info() function to understand the rows and columns I was dealing with
```
housing.info()
```
