# PROJECT 1 PROPOSAL. LEVEL 2.

# Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Importing the Data
housing = pd.DataFrame(pd.read_csv('Housing.csv'))

# The Data at a glance -- Cleaning
print(housing.describe()) # Better understanding of the Data

print(housing.head()) # This reveals a table-like format, first few rows of the data

print(housing.tail()) # Same with the previous command, only it reveals the last few rows of the data

print(housing.shape) # This reveals how many rows and columns there are

print(housing.info()) # This gives more insights on the columns I'm working with

print(housing.duplicated()) # This command checks for any duplicates in the housing dataset

print(housing.isnull().sum()) # Checking for any null values in the dataset

# Selecting the Relevant Features for Analysis -- Feature Selection/Importance
'''
This code drops the columns perceived as irrelevant features of the analysis.
With the remaining Quantitative Data,  the model can begin construction.
Encoding the Categorical Data would only set a dummy variable trap, which I would like to avoid.
'''
relevant_feats = housing.drop(
    [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'
    ], axis=1
)

y = relevant_feats.iloc[:, [0]].values # This selects the 'price' column as the Target column
X = relevant_feats.iloc[:, [1, 2, 3, 4, 5]].values # This selects the 'area' to 'parking' columns as the Features


# Exploratory Data Analysis -- Correlation Check
sns.heatmap(relevant_feats.corr(), annoy=True, cmap='coolwarm')
plt.show()

# Validation Split, Training Set and Test Set -- Model Training
'''
Here, the idea is to train the data, create a Regressor and find the best fit (model fitting),
then utilise the line on the test set created.
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression() # This variable creates a regressor
reg.fit(X_train, y_train) # This code will then 'fit the line' or, find the best fit

# Making the Prediction
y_pred = reg.predict(X_test)

# Determinining the Mean Squared Error and R2 Score -- Model Evaluation
print("Coefficients: \n", reg.coef_)
print("Variance Score: {}".format(reg.score(X_test, y_test)))

print("\nMODEL EVALUATION")
mse = round(mean_squared_error(y_test, y_pred), 2)
rs = round(r2_score(y_test, y_pred), 2)
print("Mean Squared Error: ", mse)
print("R Squared Score: ", rs)

# Plotting and Visualising the Regression -- Data Visualisation
plt.style.use('fivethirtyeight')
plt.scatter(y_pred, y_test, color='g')
plt.suptitle('Actual vs. Predicted House Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Prices')
plt.show()
      