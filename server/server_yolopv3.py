import os
import cv2
import torch
import numpy as np
import traceback

from flask import Flask, request, jsonify

# Import the Image2Mani class from the utils.image2mani module
# Импортируем класс Image2Mani из модуля utils.image2mani
from utils.image2mani import Image2Mani

# Flask application setup
# Настройка Flask-приложения
app = Flask(__name__)

# Set the maximum file upload size to 32 MB
# Устанавливаем максимальный размер загружаемого файла — 32 МБ
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

# Initialize the Image2Mani model with specific parameters
# Инициализируем модель Image2Mani с определёнными параметрами
img2main = Image2Mani(mode="tanh", speed=0.1)

# Define an API endpoint to process images
# Определяем API-эндпоинт для обработки изображений
@app.route("/process-image", methods=["POST"])
def process_image():
    # Check if an image file was provided in the request
    # Проверяем, был ли загружен файл изображения
    if "image" not in request.files:
        return jsonify({"error": "No image file provided / Файл изображения не предоставлен"}), 400

    file = request.files["image"]
    try:
        # Read and decode the image file into a NumPy array
        # Читаем и декодируем файл изображения в массив NumPy
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Validate if the image was decoded properly
        # Проверяем, было ли изображение корректно декодировано
        if image is None:
            return jsonify({"error": "Invalid image file / Неверный файл изображения"}), 400
        
        # Process the image using the Image2Mani model
        # Обрабатываем изображение с помощью модели Image2Mani
        steer, speed = img2main.run(image)
        
        # Prepare and return the response as JSON
        # Формируем и возвращаем ответ в формате JSON
        response = {"steer": steer, "speed": speed}
        return jsonify(response), 200
    
    except Exception as e:
        # Capture any exceptions and return the error traceback for debugging
        # Перехватываем исключения и возвращаем трассировку ошибки для отладки
        error_traceback = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": error_traceback}), 500

# Define a basic route for the home endpoint
# Определяем базовый маршрут для главной страницы
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "YOLOPv3 Flask server is running! / Сервер YOLOPv3 Flask работает!"})

# Run the Flask application on host 0.0.0.0 and port 5000
# Запускаем Flask-приложение на хосте 0.0.0.0 и порту 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)