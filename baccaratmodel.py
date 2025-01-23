import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate a Synthetic Dataset (Baccarat Outcomes)
np.random.seed(42)
outcomes = np.random.choice(['P', 'B', 'T'], size=1000000, p=[0.446247, 0.458597, 0.095156])  # Simulated results
data = pd.DataFrame({'Outcome': outcomes})

# 2. Create Features
def create_features(data, lookback=5):
    for i in range(1, lookback + 1):
        data[f'Prev_{i}'] = data['Outcome'].shift(i)
    data['Streak'] = (data['Outcome'] != data['Outcome'].shift(1)).cumsum()
    data['Streak_Length'] = data.groupby('Streak').cumcount() + 1
    data = data.drop('Streak', axis=1)
    return data.dropna().reset_index(drop=True)

lookback = 5
data = create_features(data, lookback)

# Encode categorical outcomes as numeric
outcome_mapping = {'P': 0, 'B': 1, 'T': 2}
for col in ['Outcome'] + [f'Prev_{i}' for i in range(1, lookback + 1)]:
    data[col] = data[col].map(outcome_mapping)

# 3. Split Dataset
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']               # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# After model training, save the model
from joblib import dump
dump(rf_model, 'baccarat_model.joblib')
dump(outcome_mapping, 'outcome_mapping.joblib')