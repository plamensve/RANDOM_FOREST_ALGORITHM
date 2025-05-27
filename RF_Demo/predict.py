import pandas as pd
import pickle

with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

new_data = pd.DataFrame([[
    12.5, 14.1, 66.0, 500.0, 0.091, 0.15, 0.2, 0.1, 0.18, 0.06,
    0.5, 1.2, 3.0, 45.0, 0.005, 0.03, 0.04, 0.02, 0.02, 0.005,
    16.0, 28.0, 100.0, 800.0, 0.12, 0.25, 0.35, 0.15, 0.25, 0.08
]], columns=[
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se",
    "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
])

prediction = model.predict(new_data)
print("Prediction:", "Malignant" if prediction[0] == 1 else "Benign")
