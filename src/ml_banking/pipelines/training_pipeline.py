# training_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

# Load the data
df = pd.read_csv("/Users/mac/Desktop/mlbanking/notebook/data/cleaned_transactions.csv")

# Separate features and labels
df_features = df.drop(columns=['TRANSACTION_ID', 'TX_FRAUD', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_FRAUD_SCENARIO'])
df_labels = df['TX_FRAUD']

# Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(df_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(standardized_features, df_labels, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Save the trained model to a file using pickle
with open('final_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the scaler for future use
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
