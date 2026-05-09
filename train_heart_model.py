import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# ✅ Ensure this file exists in the same folder
data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ✅ Save model
with open("models/heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Heart model trained and saved.")
