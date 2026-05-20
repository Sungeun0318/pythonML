from fastapi import APIRouter, Request, Response
router = APIRouter(prefix='/api/car')

@router.post("/train")
async def train(request : Request) :
    await request.json() # reqeust 객체 body 값을 직접 json 변환
    print(list)
    return True

@router.post("/predict")
async def predict(car : dict) :
    print(car)
    return 10000
