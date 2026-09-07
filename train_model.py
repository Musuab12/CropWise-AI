import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("data/Crop_recommendation.csv")

# Input features
X = data[
    ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
]

# Target: crop name
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("====================================")
print("CropWise AI - Random Forest Model")
print("====================================")
print("Dataset rows:", len(data))
print("Number of crops:", data["label"].nunique())
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save model
joblib.dump(model, "models/cropwise_rf_model.pkl")

print("Model saved successfully!")
print("====================================")