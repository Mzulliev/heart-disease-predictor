from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = 'my_key'

# загружаем модель и препроцессор
model = joblib.load('models/heart_disease_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

@app.route("/", methods=["GET"])
def home():
    lang = session.get('lang', 'ru') # по умолчанию русский
    return render_template(f'index_{lang}.html')

@app.route('/set_lang/<lang>')
def set_lang(lang):
    session['lang'] = lang
    return redirect(url_for("home"))

@app.route('/predict', methods=["POST"])
def predict():
    input_data = {
        "Age": int(request.form["Age"]),
        "Sex": request.form["Sex"],
        "ChestPainType": request.form["ChestPainType"],
        "RestingBP": int(request.form["RestingBP"]),
        "Cholesterol": int(request.form["Cholesterol"]),
        "FastingBS": int(request.form["FastingBS"]),
        "RestingECG": request.form["RestingECG"],
        "MaxHR": int(request.form["MaxHR"]),
        "ExerciseAngina": request.form["ExerciseAngina"],
        "Oldpeak": float(request.form["Oldpeak"]),
        "ST_Slope": request.form["ST_Slope"]
    }

    df = pd.DataFrame([input_data])
    X_transformed = preprocessor.transform(df)
    prediction = model.predict(X_transformed)[0]

    result_ru = "Есть риск болезни" if prediction == 1 else "Риска нет"
    result_eng = "Risk of heart disease" if prediction == 1 else "No heart disease risk"

    lang = session.get("lang", "ru")
    result = result_eng if lang == "en" else result_ru

    return render_template(f'index_{lang}.html', prediction=result)

if __name__ == "__main__":
    app.run()