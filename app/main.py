from fastapi import FastAPI
import pickle
import os
from typing import Optional
from pydantic import BaseModel
import numpy as np

app = FastAPI()
#C:\Users\Sergio\Documents\IA\deployment_ML\titanic-heroku\models

pickle_in=open("models/kaggle_titanic_model.pkl","rb")
pred=pickle.load(pickle_in)

class Passenger(BaseModel):
    Pclass: Optional[int]
    Sex: Optional[int]
    Age: Optional[float]
    SibSp: Optional[int]
    eParch: Optional[int]

@app.get("/")
def root():
    return {"message": "Api Titanic"}

@app.post(path="/predict",)

def predict(data:Passenger):
    data=data.dict()
    Pclass=data["Pclass"] 
    Age= data["Age"] 
    Sex= data["Sex"] 
    SibSp= data["SibSp"] 
    eParch= data["eParch"]
    #x_test=[3,0,0.28947368,1,1]
    #max_age=80
    x_test=[Pclass,Age/80,Sex,SibSp,eParch]
    x_test=np.reshape(x_test,(1,5))
    prediction = pred.predict(x_test)
    print("Prediccion: ",prediction) 
    if(int(prediction[0])>0.5):
        prediction="Survived"
        print("Survived")
    else:
        prediction="Do not Survived"
        print("Do not Survived")

    return {
        'prediction': prediction
    }