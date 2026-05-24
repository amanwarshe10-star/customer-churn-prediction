import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import prepare_data

# Load data
X_train, X_test, y_train, y_test, feature_names = prepare_data()

# Define model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=2.7,
    random_state=42,
    verbosity=0
)

# Train model
print("Training model...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Results
roc_auc = roc_auc_score(y_test, y_prob)
print("\n=== MODEL RESULTS ===")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
      target_names=['No Churn', 'Churn']))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as model.pkl!")