import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

print("🔍 Loading diabetes dataset...")

# Load Pima Indians Diabetes dataset from UCI (columns already clean)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
data = pd.read_csv(url, names=columns)

print("✅ Dataset loaded.")

# Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training diabetes model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model to models/ folder
os.makedirs("models", exist_ok=True)
with open("models/diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Diabetes model saved as 'models/diabetes_model.pkl'")
