import pandas as pd

#Load inbound notification pre-alert data
df = pd.read_csv('../Data/Pre-Alert.csv')

print(df.head())

# Understand the data
# Before analyzing the data, we need to consider things:
# A. What columns do we have? | B. What data types?  | C. Are there missing values? | D. How big is the dataset?

# A. Check the column names (Column inspection)
print('Columns: ')
print(df.columns)


# to be continues
