import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Generating simulated data for demonstration
np.random.seed(0)
data = {
    "Value": np.random.exponential(scale=200, size=1000),
    "Time": pd.date_range(start='1/1/2024', periods=1000, freq='H'),
    "Fraud": np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
}
df = pd.DataFrame(data)
# Histogram of transaction values
plt.figure(figsize=(10, 6))
plt.hist(df[df["Fraud"]==0]["Value"], bins=30, alpha=0.7, label='Non-Fraud')
plt.hist(df[df["Fraud"]==1]["Value"], bins=30, alpha=0.7, label='Fraud')
plt.title('Histogram of Transaction Values')
plt.xlabel('Transaction Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Time'], df['Value'], c=df['Fraud'], cmap='coolwarm', alpha=0.6)
plt.title('Transaction Value Over Time')
plt.xlabel('Time')
plt.ylabel('Transaction Value')
plt.colorbar(label='Fraud (1) / Non-Fraud (0)')
plt.show()
# Box plot
plt.figure(figsize=(10, 6))
df.boxplot(column='Value', by='Fraud')
plt.title('Box Plot of Transaction Values by Fraud Status')
plt.xlabel('Fraud Status')
plt.ylabel('Transaction Value')
plt.suptitle('')
plt.show()