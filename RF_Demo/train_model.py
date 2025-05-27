import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


data = pd.read_csv("../data/data.csv")
data = data.drop(columns=["id", "Unnamed: 32"])
data["diagnosis"] = data["diagnosis"].str.strip().map({"B": 0, "M": 1})

X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)
