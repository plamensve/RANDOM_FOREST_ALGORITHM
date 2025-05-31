import pandas as pd
import pickle

with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

radius_mean = 14.5
texture_mean = 19.2
perimeter_mean = 95.5
area_mean = 350.0
smoothness_mean = 0.1
compactness_mean = 0.15
concavity_mean = 0.12
concave_points_mean = 0.1
symmetry_mean = 0.2
fractal_dimension_mean = 0.06

radius_se = 0.5
texture_se = 1.2
perimeter_se = 3.2
area_se = 40.0
smoothness_se = 0.005
compactness_se = 0.02
concavity_se = 0.015
concave_points_se = 0.01
symmetry_se = 0.03
fractal_dimension_se = 0.004

radius_worst = 11.3
texture_worst = 25.4
perimeter_worst = 50.4
area_worst = 340.0
smoothness_worst = 0.15
compactness_worst = 0.3
concavity_worst = 0.15
concave_points_worst = 5.5
symmetry_worst = 0.02
fractal_dimension_worst = 0.05

new_data = pd.DataFrame([[
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
    fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
    smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
    fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
    smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
    symmetry_worst, fractal_dimension_worst
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
