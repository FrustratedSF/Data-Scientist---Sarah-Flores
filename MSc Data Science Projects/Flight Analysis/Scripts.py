# 1. IMPORTING LIBRARIES


import pandas as pd                #Data manipulation
import numpy as np                 # Numerical operations
import matplotlib.pyplot as plt    #plotting graphs
import seaborn as sns              #advance visuzalitions 

from scipy.stats import f_oneway   #Anova hypothesis testing

from sklearn.model_selection import train_test_split    # Splitting dataset
from sklearn.linear_model import LinearRegression       # Linear regression model
from sklearn.ensemble import RandomForestRegressor       # Random forest model

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error   #Evaluation matrix 

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
pd.set_option('display.expand_frame_repr', False)


# 2. Loading Dataset

df = pd.read_csv("Flight_dataset_4039.csv")     # Load dataset from excel


print(df.head())
print(df.info())
print(df.describe())

# 3. Data Cleaning

df = df.drop(columns = ['Unnamed: 0'], errors='ignore')     # Drop unnecessary index

print(df.isnull().sum().sort_values(ascending=False))  # Check for missing values in each column

# Fill numeric columns
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    mode_val = df[col].mode()
    df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")

# Final check
print("Total missing values: ", df.isnull().sum().sum())

# 4. Exploratory Data Analysis (EDA)

# Price Distribution
# This shows how ticket prices are distributed (skewed or normal)

plt.figure(figsize =(8,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Distribution of Ticket Prices")  # Title of plot
plt.xlabel("Price")            # X-axis label
plt.ylabel("Frequency")        # Y-axis label
plt.show()

# Skewness check (ADD HERE)
print("\nSkewness of price: ", df['price'].skew())

#Outlier detection
plt.figure(figsize=(6,4))
sns.boxplot(x=df['price'])
plt.title("Price Outliers")
plt.show()

# Price by Airline
#Compares price differences across airlines | Analyse airline effect on pricing
# Business insight: pricing strategy differences between airlines

# --- INTERPRETATION ---
# The boxplot indicates

plt.figure(figsize=(12,6))
sns.boxplot(x='airline', y='price', data=df)

plt.xticks(rotation=45)
plt.title("Price Variation by Airline")
plt.xlabel("Airline")
plt.ylabel("Price")

plt.show()

# Price vs Days Left
#Relationship between booking time and price

plt.figure(figsize=(8,5))

sns.regplot(x='days_left', y='price', data=df, scatter_kws={'alpha':0.3})

plt.title("Price vs Days Left (Booking Time)")
plt.xlabel("Days Left")
plt.ylabel("Price")

plt.show()

# --- Price by Class ---
#Compares Economy vs Business class pricing

plt.figure(figsize=(10,7))

sns.boxplot(x='class', y='price', data=df)

plt.title("Price by Travel Class")
plt.xlabel("Travel Class")
plt.ylabel("Price")

plt.show()

# 5. HYPOTHESIS TESTING 
# Tests if airlines have difference pricing
# If p < 0.05 --> significant difference

# --- ANOVA: Airline vs Price
print("H0: No difference in mean prices across airlines")
print("H1: At least one airline has different mean price")

#create groups of prices for each airline
groups = [df[df['airline'] == airline]['price'].dropna()
          for airline in df['airline'].unique()]

#Perform ANOVA test
f_stat, p_value = f_oneway(*groups)

print("F-statistic:", round(f_stat, 2))  # Shows variance between groups
print("P-value:", round(p_value, 5))  #Significance level


#Decision
if p_value < 0.05:
    print("Reject H0: Airline significantly affects price")

else:
    print("Fail to reject H0: No significant difference")


# --- Second Hypothesis
print("\nH0: No difference in price between Business and Economy class")
print("H1: Class significantly affects price")

from scipy.stats import ttest_ind

business = df[df['class'] == 'Business']['price'].dropna()
economy = df[df['class'] == 'Economy']['price'].dropna()

t_stat, p_val = ttest_ind(business, economy)

print("T-test Results")
print("T-statistic:", t_stat)
print("P-value:", p_val)

if p_val < 0.05:
    print("Reject H0: Class significantly affects price")

else:
    print("Fail to reject H0")

# 6. FEATURE ENGINEERING
#Machine learning models need numerical input converting categories into dummy variables.
#Convert categorical variables into numerical (one-hot encoding)

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
pd.set_option('display.expand_frame_repr', False)

# Identify categorical columns
print("Categorical columns:" ,df.select_dtypes(include='object').columns)

# Convert categorical variables into numerical 
df = df.drop('flight', axis=1)

df_encoded = pd.get_dummies(df, drop_first=True)

#Convert boolean columns to integers (True/False -> 1/0)
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)


# Compare dataset shape before and after encoding
print("Original shape:", df.shape)
print("Encoded shape:", df_encoded.shape)
print(df_encoded.head())

# 7. DEFINE FEATURES AND TARGET

X = df_encoded.drop('price', axis=1)  #Independent variables (features)
y = df_encoded['price']        #Dependent variable (target)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# 8. TRAIN-TEST SPLIT

#Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# 9. MODEL 1: LINEAR REGRESSION

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import numpy as np


lr = LinearRegression()  #Create Linear regression model
lr.fit(X_train, y_train)  #Train model on training data

y_pred_lr = lr.predict(X_test)  #Predict prices on test data

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)

print("Linear Regression Results:")
print("R2 Score:", r2_lr)
print("RMSE:", rmse_lr)
print("MAPE:", mape_lr)

# 10. MODEL 2 : RANDOM FOREST REGRESSION

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=20, random_state=42)  # Create model with 20 trees
rf.fit(X_train, y_train)  #Train mode

y_pred_rf = rf.predict(X_test)  # Predict prices

print("First 10 predictions:", y_pred_rf[:10])

# 11. MODEL EVALUATION FUNCTION

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import numpy as np

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)   #Measures accuracy (higher is better)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))   #Measures prediction error
    mape = mean_absolute_percentage_error(y_true, y_pred)   #Percentage error

    return {"R2": r2, "RMSE": rmse, "MAPE": mape}


# EVALUATE LINEAR REGRESSION
lr_results = evaluate_model(y_test, y_pred_lr)

print("\nLinear Regression Results:")
print(lr_results)

# EVALUATE RANDOM FOREST
rf_results = evaluate_model(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print(rf_results)


#Insights: R2 --> accuracy (higher = better)
# RMSE --> error magnitude (lower = better)
# MAPE --> % error

# 12. FEATURE IMPORTANCE (RANDOM FOREST)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

importances = rf.feature_importances_  #Extrat importance of each feature

# Creat DataFrame for better visualisation

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df.head(10))  #Show top 10 important features

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title("Top 10 Important Features Affecting Price")
plt.show()

# 13. FINAL INTERPRETATION (PRINT INSIGHTS)

print("\nBusiness Insights:")
print("- Ticket prices increase as travel duration increases.")
print("- Business class significantly increases ticket prices.")
print("- Booking earlier (more days left) generally reduces ticket cost.")
print("- Airline choice has a statistically significant impact on pricing.")
print("- Random Forest model performas better due to capturing non-linear relationshops.")
print("- Airlines can optimize pricing strategies based on demand patterns.")
print("- Customers can save costs by booking tickets earlier.")
