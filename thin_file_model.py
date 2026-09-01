import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate Complete Multi-Source Alternative Dataset
np.random.seed(42)
n_samples = 1000

data = {
    'Account_Age_Months': np.random.randint(1, 36, n_samples),          # Telecom account age
    'Monthly_Recharge_Avg': np.random.uniform(100, 1500, n_samples),     # Telecom spending
    'Recharge_Delay_Days': np.random.randint(0, 15, n_samples),          # Telecom recharge delay days
    'Data_Usage_Stability': np.random.uniform(0.1, 1.0, n_samples),      # Mobile data consistency
    'Utility_OnTime_Payment_Rate': np.random.uniform(0.3, 1.0, n_samples), # Electricity/Water bill consistency
    'UPI_Monthly_Txn_Count': np.random.randint(2, 50, n_samples)         # Frequency of digital cash flow / UPI transactions
}

df = pd.DataFrame(data)

# Target Variable: 1 = Good Credit Risk (Eligible), 0 = High Risk
# Comprehensive logic combining telecom stability, utility bills, and UPI activity
df['Credit_Eligible'] = np.where(
    (df['Recharge_Delay_Days'] < 5) & 
    (df['Account_Age_Months'] > 6) & 
    (df['Utility_OnTime_Payment_Rate'] > 0.6) & 
    (df['UPI_Monthly_Txn_Count'] > 8), 1, 0
)

# 2. Split Data into Features (X) and Target (y)
X = df[['Account_Age_Months', 'Monthly_Recharge_Avg', 'Recharge_Delay_Days', 
        'Data_Usage_Stability', 'Utility_OnTime_Payment_Rate', 'UPI_Monthly_Txn_Count']]
y = df['Credit_Eligible']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate Performance
y_pred = model.predict(X_test)
print("--- Comprehensive Multi-Source Model Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("COMPLETE MODEL TRAINING SUCCESSFUL!")
