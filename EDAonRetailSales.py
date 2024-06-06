# PROJECT 1 PROPOSAL. LEVEL 1.

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
