"""
Импортируем необходимые библиотеки
"""
import joblib
from io import StringIO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from fastapi import HTTPException
import re

app = FastAPI()
"""
Загружаем обученную модель и скалер
"""
model = joblib.load("model2.pkl")
scaler = joblib.load("scaler_lin.pkl")

"""
Это класс - 1 автомобиль со следующими признаками:
"""
class Item(BaseModel):
    year: float
    km_driven: float
    mileage: float
    engine: float
    max_power: float
    seats: float

"""
А это класс со списком автомобилей со следующими признаками:
"""
class Items(BaseModel):
    objects: List[Item]

"""
Тут мы принимаем 1 объект, получаем признаким и нормализуем их
"""
def preprocess_item(item: Item) -> np.array:

    features = [
      item.year,
      item.km_driven,
      item.mileage,
      item.engine,
      item.max_power,
      item.seats
    ]

    features_scaled = scaler.transform([features]) 
    return features_scaled

"""
Функция для предсказания стоимости авто:
"""
def predict_car_price(item: Item) -> float:
    try:    
        features_scaled = preprocess_item(item)
        predicted_price = model.predict(features_scaled)[0]
        return predicted_price
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
Эндпоинт для предсказания цены одного автомобиля:
"""
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    predicted_price = predict_car_price(item)
    return predicted_price
"""
Эндпоинт для предсказания цены списка авто :
"""
@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    predicted_prices = [predict_car_price(item) for item in items]
    return predicted_prices

"""
Эндпоинт для предсказания цены списка авто через csv файл:
"""
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)) -> dict:
    try:
        data = await file.read() # Чтение содержимого файла
        data_string = data.decode('utf-8') 
        csv_file = StringIO(data_string)
        
        df = pd.read_csv(csv_file) # CSV в DataFrame

        predicted_prices = []
        
        # Извлекаем признаки
        for i, row in df.iterrows():

            item = Item(
                year=row['year'],
                km_driven=row['km_driven'],
                mileage=row['mileage'],
                engine=row['engine'],
                max_power=row['max_power'],
                seats=row['seats']
            )

            predicted_price = predict_car_price(item)
            predicted_prices.append(predicted_price)

        df['predicted_price'] = predicted_prices
        # Сохраняем в файл
        output_file = "predicted_prices.csv"
        df.to_csv(output_file, index=False)
        return FileResponse(output_file, media_type='csv', filename="predicted_prices.csv")
    except Exception as e:
        return {"error": str(e)}