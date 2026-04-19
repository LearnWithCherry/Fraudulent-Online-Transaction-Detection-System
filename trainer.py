import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score

# 1. Generate Dataset
n = 10000
data = pd.DataFrame({
    'amt': np.random.uniform(100, 250000, n),
    'avg_amt': np.random.uniform(500, 100000, n),
    'tx_count': np.random.randint(1, 500, n),
    'dist': np.random.uniform(0, 1000, n),
    'origin': np.random.randint(0, 2, n)
})

# 2. Logic: Define Fraud
condition = ((data['amt'] > (data['avg_amt'] * 3)) | ((data['origin'] == 1) & (data['dist'] > 800)))
data['label'] = condition.astype(int)
# 3. Split & Train
X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 4. Metrics (CORRECTED)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

# Export proofs
proofs = {
    "training_accuracy": float(train_acc),
    "validation_accuracy": float(test_acc),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "confusion_matrix": cm.tolist(),
    "samples": [{"actual": int(a), "predicted": int(p)} for a, p in zip(y_test[:5], y_test_pred[:5])]
}

with open('ml_proofs.json', 'w') as f:
    json.dump(proofs, f)

joblib.dump(model, 'sentinel_model.pkl')

print("Training Accuracy:", train_acc)
print("Validation Accuracy:", test_acc)
print("✅ sentinel_model.pkl AND ml_proofs.json CREATED.")