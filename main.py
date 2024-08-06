import json

import dill
import logging

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

app = FastAPI()
with open('/home/alexander/ds-intro/34_Finals/model.pkl', 'rb') as file:
   model = dill.load(file)
   logging.info('model loaded successfully')

# logging.basicConfig(level=logging.DEBUG)  # Устанавливаем уровень логирования


class Form(BaseModel):
    session_id:Union[str,None] = None
    client_id:Union[str,None] = None
    visit_date:Union[str,None] = None
    visit_time:Union[str,None] = None
    visit_number:Union[int,None] = None
    utm_source:Union[str,None] = None
    utm_medium:Union[str,None] = None
    utm_campaign:Union[str,None] = None
    utm_adcontent:Union[str,None] = None
    utm_keyword:Union[str,None] = None
    device_category:Union[str,None] = None
    device_os:Union[str,None] = None
    device_brand:Union[str,None] = None
    device_model:Union[str,None] = None
    device_screen_resolution:Union[str,None] = None
    device_browser: Union[str,None] = None
    geo_country: Union[str,None] = None
    geo_city: Union[str,None] = None
    
    

class Prediction(BaseModel):
    target: bool


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    # app.logger.debug('Получен POST запрос на /predict')
    # app.logger.debug(f'Данные из запроса: {request.json}')
    # print(form.dict())
    
    df = pd.DataFrame.from_dict([form.dict()])
    print('dataframe ',df)
    y = model['model'].predict(df)
    print("Prediction result:", y) 

    return {
       'id': form.id,
       'price_category': y[0]
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, debug=True)
