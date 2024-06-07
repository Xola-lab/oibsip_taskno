# Projects Overview
#### There are 3 projects under this repo:
- Predicting House Prices using Linear Regression
- Wine Quality Prediction
- Exploratory Data Analysis (EDA) on Retail Sales Data

#### References
Books:
- Machine Learning with Python
- Practical Statistics for Data Scientists

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
```python
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
```python
housing = pd.DataFrame(pd.read_csv("Housing.csv")
```

To get a glance at the data, I performed some simple commands:
- I looked at the info() function to understand the rows and columns I was dealing with
```python
housing.info()
```

<p align="center"> 
<img src="1 - House Price Prediction (Files)/EDA RESULTS/SHAPE AND INFO RESULT.png">
</p>

There were 545 rows and 13 columns.

- I then checked if the data had any duplicated values
```python
housing.duplicated()
```
There  were no duplicates in the data.

<p align="center"> 
<img src="1 - House Price Prediction (Files)/EDA RESULTS/DUPLICATES CHECK - RESULT.png">
</p>

- I also checked for any null values in the dataset
```python
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
```python
relevant_feats = housing.drop(
    [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'
    ], axis=1
)
```
The data being dropped here are the columns containing Categorical data (e.g., "Yes", "No", etc.).
The only data remaining now was the relevant features needed for the analysis.

Then, having removed the Categorical data, I set the Target (y) and Features (X) variables:
```python
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
```python
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
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
```

### Prediction
After the model had been trained and a fitting done, the next step was to make a prediction.
For making the prediction, I used the following code:
```python
y_pred = reg.predict(X_test)
```

### Model Evaluation 
The next step in the process was to evaluate the performance of the model:
```python
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
```python
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

Athough in the correlation analysis there was a weak correlation, the linear regression model still accounted for these factors and still found a linear relationship. Also, the size of the sample played a role in this result: 80% of the Housing data was used for training the model, whilst the remaining 20% was used for testing the model. Another thing to consider is that Linear Regression assumes a linear relationship between the Target variable and the Features.

Other conditions of the model have been met, despite the weak correlation; and therefore, the model still produced valid results.

### Summary:
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

### Library and Data Import
The first thing here is to import all the required libraries:
```python
# Library Import
'''Dealing with the raw data'''
import pandas as pd

'''Data Visualisation'''
import matplotlib.pyplot as plt
import seaborn as sns

'''Model Training'''
from sklearn.model_selection import train_test_split

'''The Classifier Models'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

'''Model Evaluation'''
from sklearn.metrics import accuracy_score, classification_report as report
```

Then, loading the Data:
```python
# Data Import
wine = pd.DataFrame(pd.read_csv("WineQT.csv"))
```

### Data Cleaning
These next steps are not dissimilar from the first project. Here, I inspected the data first and checked for any duplicates or null values.

```python
'''Cleaning'''
print(wine.duplicated())
print(wine.isnull().sum())
```

Fortunately, this dataset also had neither Duplicates nor Null Values:

<p align="center"> 
<img src="2 - Wine Prediction (Files)/3b - DATA INSPECTION - DUPLICATES RESULTS.png">
</p>

<p align="center"> 
<img src="2 - Wine Prediction (Files)/3a  - DATA INSPECTION -- NULL VALUE RESULT.png">
</p>

### Feature Selection
Again, it was imperative to select the relevant features for the models:

```python
# (Chemical Qualities) Selecting Density and Acidity -- Feature Selection
'''First, we want to drop all characteristics that are deemed irrelevant'''
chemical_chars = wine.drop(
    [
        "residual sugar", "chlorides", "free sulfur dioxide", "sulphates", "Id"
    ], axis=1
)

print(chemical_chars.head())

'''Next, we want to determine the Target and Feature variables'''
X = chemical_chars.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values # This selects our Feature variables
y = chemical_chars.iloc[:, [7]].values # This selects our Target variable -- The 'price' column
```

Having dropped these other features, the remaining features were as follows:

<p align="center"> 
<img src="2 - Wine Prediction (Files)/4a - FEATURE SELECTION - RESULT.png">
</p>

### Data Preprocessing
With the relevant data selected, it was now time to explore the data by checking Correlation and Distribution:

```python
# Exploratory Data Analysis (EDA) -- Data Preprocessing
'''Correlation Check'''
sns.heatmap(chemical_chars.corr(), annot=True, cmap='coolwarm')
plt.show()

'''Distribution Analysis'''
sns.pairplot(chemical_chars)
plt.show()
```

This was the result for the Correlation Check:

<p align="center">
<img src="2 - Wine Prediction (Files)/5a - DATA PREPROCESSING - HEAT MAP.png">
</p>

As can be seen, some of the chemical characteristics have a negative correlation. What this mean is, when the quality increases, the chemicals effect decreases -- and vice versa.
The other features do have a correlation, but it's not strong enough. However, this isn't enough to disregard these features yet -- "Correlation does not mean Causation".

As for the Distribution:

<p align="center">
<img src="2 - Wine Prediction (Files)/5b - DATA PREPROCESSING - PAIRPLOT.png">
</p>

Essentially, what this means/depicts is the Distribution of each quality with each other.

### Model Training (Classifier Models)
This step is similar to the first project and, as you'll see, the only difference is that there is more than one Model -- There are three models, each simple to initiate.
However, it would be remiss not to split the data before model selection, so the first step was to split the data and then train the three models as well as fit the models to the data and, lastly, making the predictions and providing the accuracy scores:

```python
# Validation Split: Training and Test Sets -- Model Training/Classifier Models
'''Splitting the Data'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''Model Selection 1 - Random Forest'''
rand_f = RandomForestClassifier(n_estimators=100, random_state=42)
rand_f.fit(X_train, y_train) # Model Fitting

rand_f_pred = rand_f.predict(X_test) # Prediction
rand_f_acc = accuracy_score(y_test, rand_f_pred) # Accuracy

'''Model Selection 2 - Stochastic Gradient Descent'''
sgd_mod = SGDClassifier(random_state=42)
sgd_mod.fit(X_train, y_train)

sgd_pred = sgd_mod.predict(X_test)
sgd_acc = accuracy_score(y_test, sgd_pred)

'''Model Selection 3 - Support Vector Classifier'''
svc_mod = SVC(kernel='linear')
svc_mod.fit(X_train, y_train)

svc_pred = svc_mod.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)
```

### Model Performance
Having trained the models, the next step in the process is to provide each model's performance:

```python
# Performance Check -- Model Evaluation
print("MODEL EVALUATION")
'''Model 1 Evaluation'''
print("RANDOM FOREST")
print("Random Forest Accuracy: ", rand_f_acc)
print("Report: ", report(y_test, rand_f_pred))

'''Model 2 Evaluation'''
print("STOCHASTIC GRADIENT DESCENT")
print("SGD Accuracy: ", sgd_acc)
print("SGD Report: ", report(y_test, sgd_pred))

'''Model 3 Evaluation'''
print("SUPPORT VECTOR CLASSIFIER")
print("SVC Accuracy: ", svc_acc)
print("SVC Report: ", report(y_test, svc_pred))
```

Output:

Random Forest:
<p align="center">
<img src="2 - Wine Prediction (Files)/7a - MODEL EVALUATION - RANDOM FOREST.png">
</p>

Stochastic Gradient Descent:
<p align="center">
<img src="2 - Wine Prediction (Files)/7b - MODEL EVALUATION - SGD.png">
</p>
<p align="center">
<img src="2 - Wine Prediction (Files)/7bb - MODEL EVALUATION - SDG.png">
</p>

Support Vector Classifier:
<p align="center">
<img src="2 - Wine Prediction (Files)/7c - MODEL EVALUATION - SVC.png">
</p>
<p align="center">
<img src="2 - Wine Prediction (Files)/7cc - MODEL EVALUATION - SVC.png">
</p>

If these evaluations do not make sense at this point, fret not, the last part of the project will clear it up.

### Data Visualisation
This final step involves visualising the accuracy each of these models:

```python
# Plotting and Visualisation -- Data Visualisation
models = ["Random Forest Model", "Stochastic Gradient Descent", "Support Vector Classifier"]
accuracy = [rand_f_acc, sgd_acc, svc_acc]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracy, color=['g', 'r', 'blue'])
plt.xlabel("Models")
plt.ylabel("Model Accuracy")
plt.title("Accuaracy vs. Model")
plt.ylim(0, 1)
plt.show()
```

Here, with the help of Python Lists, two variables were created; a 'models' variable to fit the names of the models and an 'accuracy' variable to fit the accuracies of each model.
Then, the bar graph utilising Matplotlab. The outcome was as follows:

<p align="center">
<img src="2 - Wine Prediction (Files)/8a - DATA VIZ. - RESULT.png">
</p>

Now, it becomes clear what the accuracy of each model is. Clearly, the Random Forest Model is a much better model in this case, with the quality of wine nearly at 66% accuracy.

### Summary
This project was not different from the first, it involved largely the same process (with a little difference in the number of models used). The process involved in this project were Importing the relevant Libraries, Data Inspection (getting a glimpse of the data), Checking for Duplicates and Null Values, Feature Selection, Data Preprocessing, Training the different models, Evaluating these models and, finally, Visualising the results.

## 3 - Exploratory Data Analysis (EDA) on Retail Sales Data
(Project 1 Proposal, Level 1)

### Overview
The objective here is to perform Exploratory Data Analysis (EDA) to uncover Patterns, Trends, and Insights that can be helpful in making informed business decisions.

### Data Sources
Kaggle: The data for this last prpject was also collected from Kaggle.

### Tools
Python -- EDA, Data Visualisation
- Pandas
- Matplotlib
- Seaborn

### Library Import, Data Import, and Data Inspection
```python
# Library Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Import
retail = pd.DataFrame(pd.read_csv("retail_sales_dataset.csv"))

# Data Inspection
'''The first 10 rows associated with the Data'''
print(retail.head(10))

'''Summary Statistics of the Data'''
print(retail.describe())

'''Checking and Handling Missing Values'''
print(retail.isnull().sum())
```

Output:

<p align="center">
<img src="3 - EDA on Retail Sails Data (Files)/EXPLORATION.png">
</p>

The columns associated with this dataset are seven and the data has no missing values.

### Sales Over Time Analysis

```python

# Sales Trend over time
'''Converting the 'Date' column to datetime format'''
retail['Date'] = pd.to_datetime(retail['Date'])

'''Setting the 'Date' column as an index'''
retail.set_index('Date', inplace=True)

'''Determining the Monthly Sales'''
sales_per_month = retail.resample('ME').sum()

'''Plotting Sales over Time'''
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 6))
plt.plot(sales_per_month.index, sales_per_month['Total Amount'])
plt.title("Monthly Sales")
plt.xlabel("Period - Year/Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()
```

Output:

<p align="center">
<img src="3 - EDA on Retail Sails Data (Files)/SALES OVER TIME.png">
</p>

Here, observation led me to the analysis that from February to December, the total spending was in a 'ranging' trend -- going up and down in a horizontal (or 'forward') direction with total spending ranging between 2000 to slightly above 5000. Then, in January, the spending impulsively dropped; indicating a lack of spending by the customers.

### Customer Demographics and Customer Purchasing Behaviour

```python
# Customer and Product Analysis
'''CUSTOMER DEMOGRAPHICS'''
gender_dist = retail['Gender'].value_counts() # Gender distribution

age_dist = retail['Age'].value_counts() # Age distribution

gender_age = retail.groupby(['Gender', 'Age']).size().unstack() # Purchasing behaviour based on Gender and Age

'''Plotting Customer Demographics'''
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
gender_dist.plot(kind='bar', color=['r', 'blue'])
plt.title("Customer Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.subplot(1, 2, 2)
age_dist.sort_index().plot(kind='bar', color='g')
plt.title("Customer Distribution by Age")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout(pad=3)
plt.show()

'''PURCHASING BEHAVIOUR'''
'''Total Spending Distribution'''
plt.style.use('fivethirtyeight')
sns.displot(retail['Total Amount']) # Selecting the 'Total Amount' column
plt.title("Distribution of Total Amount Spent")
plt.xlabel("Total Amount")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

'''Average Spending'''
avg_spending_per_transaction = retail.groupby('Customer ID')['Total Amount'].mean().mean()

'''Purchase Frequency'''
purchases = retail['Customer ID'].value_counts().value_counts().sort_index()
print("Purchasing Frequency: ", purchases)
print("\nAverage Spending per Transaction: ${:.2f}".format(avg_spending_per_transaction))
```

Output:
#### Customer Demographics
<p align="center">
<img src="3 - EDA on Retail Sails Data (Files)/CUSTOMER DEMOGRAPHICS.png">
</p>

In the first case, if one pays attention to the Gender Distribution, it can be seen that there isn't much difference in the number of Males and Females -- although, the females do have a slight lead in numbers.

Secondly, paying attention to the Age Distribution, we can see that people within the ages 22, 34, 43, 51, 54, 57, 62, and 64 have the highest number of customers whilst 33 and 58 year-olds have the lowest number of customers.

Regardless, the data suggests that there are a high number of customers from almost all age groups -- excluding 70 to 80 year-olds.

#### Customer Purchasing Behaviour

<p align="center">
<img src="3 - EDA on Retail Sails Data (Files)/TOTAL SPENDING DISTRIBUTION.png">
</p>

In in this case, we can see that customers who spend $0 - $200 have a high frequency indicating a high number of customer spending within the range. The rest of the spending has a low frequency: spending in the $800 range is lower compared with the $200 range, this is the same with spwnding in the $1000 - $1500 and $1800 - $2000 ranges.

From this, we can make an elementary deduction: most items customers spend their money on are below $200, the secondary items they spend on are below $800. All the other items ranging between $1000 and $2000 have a low frequency of being spent on -- this can be attributed to several economic factors such as poverty.

#### Average Spending

```
Average Spending per Transaction: $456.00
```

This, of course, should be clear enough: the average spending per transaction by the customers is $456.00.

### Summary
It is surely recommended that Business looking at this analysis should produce more items in the period between March and November, maybe even more in June. Additionally, Businesses should reduce production in January.

Businesses should seek to produce more items that meet the level of living of the people, taking into consideration economic factors.

## Conclusion
These three projects have been insightful to me, in my journey to uncovering the insights contained in all the data, I stumbled upon a couple of problems -- problems that required a little bit of research before engagement.

Consider the following: the Analysis process involved in these projects in not dissimilar to a string, a thread -- with a beginning an end. This thread sometimes broke and, with a bit of logic and deduction (not forgetting a little bit domain knowledge), I was able to tie these points where the thread broke and when that didn't help I had pull some new thread and make new connections.

This 'thread' began with getting the data inspected under a lense, then selecting 'features if interest' -- parts of the data that was relevant. In two of the cases, these features of interest had to be trained to create predictive models.

In my intimacy with this thread I was able to discover 'hidden meanings' in the data and arriving at insightful conclusions, thereby solving the mystery of it all -- turning raw data to digestible pieces.

I look forward to picking up the threads of this tangled web of Data Analytics - THANK YOU!






