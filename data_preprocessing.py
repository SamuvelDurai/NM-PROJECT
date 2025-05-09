import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("creditcard.csv")

# 1. Basic Info
print("✅ Dataset Loaded")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Null Check
print("\n🧼 Missing Values:\n", df.isnull().sum())

# 3. Data Types
print("\n📋 Data Types:\n", df.dtypes)

# 4. Class Distribution
print("\n⚖️ Class Distribution:\n", df['Class'].value_counts())

# 5. Feature Engineering
# Log-transform the 'Amount' feature
df['Amount_log'] = np.log1p(df['Amount'])

# Scaling 'Amount' and 'Time'
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])

# Drop original 'Amount' and 'Time'
df_cleaned = df.drop(['Amount', 'Time'], axis=1)

# 6. Save the cleaned data
df_cleaned.to_csv("creditcard_cleaned.csv", index=False)
print("\n📁 Cleaned data saved as 'creditcard_cleaned.csv'")

# 7. Optional Visualizations
# Class Distribution
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()

# Correlation with Target
corr = df_cleaned.corr()['Class'].sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=corr.values, y=corr.index)
plt.title("Correlation with Fraud Class")
plt.show()
