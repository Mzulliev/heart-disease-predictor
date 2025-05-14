import joblib
import pandas as pd


model = joblib.load("models/heart_disease_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

sample = pd.DataFrame([{
    "Age": 70,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 160,
    "Cholesterol": 320,
    "FastingBS": 1,
    "RestingECG": "LVH",
    "MaxHR": 100,
    "ExerciseAngina": "Y",
    "Oldpeak": 3.0,
    "ST_Slope": "Flat"
}])

X_transformed = preprocessor.transform(sample)

prediction = model.predict(X_transformed)

print(f'Предсказание (0 = нет болезни, 1 = болезнь): {prediction[0]}')