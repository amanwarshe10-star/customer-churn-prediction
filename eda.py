import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Basic info
print("=== DATASET INFO ===")
print("Total customers:", df.shape[0])
print("Total columns:", df.shape[1])
print("\nFirst 5 rows:")
print(df.head())

print("\n=== CHURN COUNT ===")
print(df['Churn'].value_counts())

# Graph 1 - Churn Distribution
plt.figure(figsize=(5,4))
df['Churn'].value_counts().plot(kind='bar', color=['green','red'])
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.show()
print("Graph 1 saved!")

# Graph 2 - Contract vs Churn
plt.figure(figsize=(7,4))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Contract Type vs Churn')
plt.tight_layout()
plt.savefig('contract_churn.png')
plt.show()
print("Graph 2 saved!")

print("\n=== EDA Complete! ===")