from fastapi import FastAPI
import uvicorn
import controller

app = FastAPI() 

app.include_router( controller.router )

if __name__ == "__main__" :
    uvicorn.run( 'app:app' , host='127.0.0.1' , port = 8000 , reload=True )

