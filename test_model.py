import joblib
import pandas as pd


model = joblib.load("models/heart_disease_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

sample = pd.DataFrame([{
    "Age": 39,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 130,
    "Cholesterol": 200,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 190,
    "ExerciseAngina": "N",
    "Oldpeak": 0.0,
    "ST_Slope": "Up"
}])

X_transformed = preprocessor.transform(sample)

prediction = model.predict(X_transformed)
predction_proba = model.predict_proba(X_transformed)

print(f'Предсказание (0 = нет болезни, 1 = болезнь): {prediction[0]}\n\
      Вероятность: {predction_proba[0][prediction[0]]}')