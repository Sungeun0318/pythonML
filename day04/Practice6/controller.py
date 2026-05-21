from fastapi import APIRouter, Request, Response
router = APIRouter(prefix='/api/car')
from service import car_service

@router.post("/train")
# http://localhost:8080/api/car/train
async def train(request : Request) :
    list = await request.json() # reqeust 객체 body 값을 직접 json 변환
    return car_service.train(list)

@router.post("/predict")
# http://localhost:8080/api/car/predict
async def predict(car : dict) :
    return car_service.predict(car)
