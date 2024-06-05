# PROJECT 2 PROPOSAL. LEVEL-2.

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

# Data Import
wine = pd.DataFrame(pd.read_csv("WineQT.csv"))

# Data Inspection
'''Observation'''
print(wine.describe())
print(wine.head())
print(wine.tail())
print(wine.info())
print(wine.shape)

'''Cleaning'''
print(wine.duplicated())
print(wine.isnull().sum())

# (Chemical Qualities) Selecting Density and Acidity -- Feature Selection
'''First, we want to drop all characteristics that are deemed irrelevant'''
chemical_chars = wine.drop(
    [
        "residual sugar", "chlorides", "free sulfur dioxide", "sulphates", "Id"
    ], axis=1
)

print(chemical_chars.head())

'''Next, we want to dtermine the Target and Feature variables'''
X = chemical_chars.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values # This selects our Feature variables
y = chemical_chars.iloc[:, [7]].values # This selects our Target variable -- The 'price' column

# Exploratory Data Analysis (EDA) -- Data Preprocessing
'''Correlation Check'''
sns.heatmap(chemical_chars.corr(), annot=True, cmap='coolwarm')
plt.show()

'''Distribution Analysis'''
sns.pairplot(chemical_chars)
plt.show()

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
