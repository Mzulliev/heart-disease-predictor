# Heart Disease Risk Predictor 🫀

Простое веб-приложение для предсказания риска сердечно-сосудистых заболеваний на основе пользовательского ввода. Использует обученную модель машинного обучения и работает через Flask.

## 🚀 Возможности

- Веб-интерфейс на Flask
- Предсказание наличия болезни по 11 признакам
- Двухъязычная поддержка: 🇷🇺 Русский / 🇬🇧 English
- Подключён обученный пайплайн: препроцессинг + модель

## 🧠 Используемые технологии

- Python 3.10+
- Flask
- Pandas, Scikit-learn
- HTML + CSS


## 🛠️ Установка и запуск

1. Клонируй репозиторий:
    ```bash
   git clone https://github.com/Mzulliev/heart-disease-predictor.git
   cd heart-disease-predictor

2. Создай виртуальное окружение и активируй его:
    ``` bash
    python -m venv venv
    source venv/bin/activate        # Для Linux/macOS
    venv\Scripts\activate           # Для Windows
3. Установка зависимостей
    ```bash
    pip install -r requirements.txt
4. Запусти приложение
    ```bash
    python main.py
5. Открой в браузере
    ```bash
    http://127.0.0.1:5000

## 🌱 План развития
-	🔁 Перевод проекта на FastAPI
-	🐳 Обернуть проект в Docker-контейнер
-	🌍 Развертывание на бесплатном сервере (например, Render, Railway)
-	✅ Добавление тестов
-	🧪 Использование CI/CD для автообновлений

## Автор
- GitHub: [mzulliev](https://github.com/Mzulliev)
- Telegram: [@mzulliev](https://t.me/mzulliev)

📌 Проект создан с целью обучения и демонстрации полного цикла ML-разработки.

## Данные
Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset