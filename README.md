# Projects Overview
There are 3 projects under this repo:
- Predicting House Prices using Linear Regression
- Wine Quality Prediction
- Google Playstore Data Analysis

## 1 - Predicting House Prices with Linear Regression
(Project 1 Proposal. Level 2.)

### Overview
The main goal of this project is to build a predictive model using Multiple Linear Regression to estimate house prices based on certain features.

### Data Sources
Kaggle Dataset: The project utilises a publicly available housing dataset found on Kaggle ("Housing.csv").

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
- I looked at the info() function to understand the rows and columns I was dealing with
```
housing.info()
```

<p align="center"> 
<img src="1 - House Price Prediction (Files)/EDA RESULTS/SHAPE AND INFO RESULT.png">
</p>

There were 545 rows and 13 columns.

- I then checked if the data had any duplicated values
```
housing.duplicated()
```
There duplicates in the data.

<p align="center"> 
<img src="1 - House Price Prediction (Files)/EDA RESULTS/DUPLICATES CHECK - RESULT.png">
</p>

- I also checked for any null values in the dataset
```
housing.isnull().sum()
```
There were no missing values in the data:

<p align="center"> 
<img src="1 - House Price Prediction (Files)/EDA RESULTS/NULL VALUE CHECK - RESULT.png">
</p>

The Data did not contain any duplicated values or any null values.

### Feature Selection
The data contains Quantitative and Categorical Data. To make the model, we need only the Quantitative data. 
So, I removed the Categorical data in the dataset with the following code:
```
relevant_feats = housing.drop(
    [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'
    ], axis=1
)
```
The data being dropped here are the columns containing Categorical data (e.g., "Yes", "No", etc.).
The only data remaining now was the relevant features needed for the analysis.

Then, having removed the Categorical data, I set the Target (y) and Features (X) variables:
```
y = relevant_feats.iloc[:, [0]].values # The 'price' column -- Target.
X = relevant feats.iloc[:, [1, 2, 3, 4, 5]].values # The 'area' to 'parking' columns -- Features.
```

The result of the above code was as follows:

<p align="center"> 
<img src="1 - House Price Prediction (Files)/4a - FEATURE SELECTION - RESULT.png">
</p>

Only 6 columns were chosen, 'price' being the target and the rest of the columns as the 'features'.

### Exploratory Data Analysis 
The next thing I did was do a correlation check on the remaining data, I did this by utilising Seaborn's heat map:
```
sns.heatmap(relevant_feats.corr(), annot=True, cmap='coolwarm')
plot.show()
```

The result was as follows:

<p align="center">
<IMG
SRC="1 - House Price Prediction (Files)/5 - CORRELATION CHECK - RESULT.jpg"
</p>

Although the correlation had a relatively weak linear relationship (less than 0.6), the changes in these features are not strongly reflected in the changes in Price. This correlation does not precisely capture all the relationships, and there are meaningful connections that are not exactly linear. For example, in the real estate world, one may find that houses with more bedrooms normally have high prices -- this is the same with the rest of the features.

### Model Training
This process involved splitting the data into a Training and a Test Set, creating a Regressor, and fitting the model:
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
```

### Prediction
After the model had been trained and a fitting done, the next step was to make a prediction.
For making the prediction, I used the following code:
```
y_pred = reg.predict(X_test)
```

### Model Evaluation 
The next step in the process was to evaluate the performance of the model:
```
print("Coefficients: \n", reg.coef_)
print("Variance Score: {}".format(reg.score(X_test, y_test)))

print("\nMODEL EVALUATION")
mse = round(mean_squared_error(y_test, y_pred), 2)
rs = round(r2_score(y_test, y_pred), 2)
print("Mean Squared Error: ", mse)
print("R Squared Score: ", rs)
```

The result looked like this:

<p align="center">
<IMG
SRC="1 - House Price Prediction (Files)/8a - MODEL EVALUATION - RESULT.png"
</p>

But what does this mean? Firstly, the variance score here means that nearly 54.6% of the variance in the target (Price) can be explained by the features (area, bedrooms, bathrooms, and parking). Secondly, this compliments the R-squared score of 0.55, which indicates that the model only explains a small amount of the variability in the price. However, given the correlation analysis and the fact that these features do not strongly reflect a linear relationship, we can rest assured that the model can still produce valid results -- although there is room for some moderation (and improvement).

### Data Visualisation 
The last and final step was to plot the regression:
```
plt.style.use('fivethirtyeight')
plt.scatter(y_pred, y_test, color='g')
plt.suptitle('Actual vs. Predicted House Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Prices')
plt.show()
```
This was the resulting graph:

<p align="center">
<IMG
SRC="1 - House Price Prediction (Files)/9a - DATA VIZ. - RESULT.png"
</p>

Athough in the correlation analysis there wasn't a weak correlation, the linear regression model still accounted for these factors and still found a linear relationship. Also, the size of the sample played a role in this result: 80% of the Housing data was used for training the model, whilst the remaining 20% was used for testing the model. Another thing to consider is that Linear Regression assumes a linear relationship between the Target variable and the Features.

Other conditions of the model have been met, despite the weak correlation; and therefore, the model still produced valid results.

#### Summary:
This project involved Exploring and Cleaning the data, Feature Selection, Model Training and Evaluation, Making the Prediction, and finally, Visualisation.

## 2 - Wine Quality Prediction
(Project 2 Proposal. Level 2.)

### Overview
The main objective of this project is to predict the quality of wine based on certain chemical characteristics, namely 'Density' and 'Acidity'.

### Data Sources
Kaggle Dataset: This project also uses a publicly available dataset found on Kaggle.

### Tools
Python - EDA, Cleaning, Data Analysis, and Data Visualisation.
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Sci-kit Learn
