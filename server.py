from typing import Union
from fastapi import FastAPI
import uvicorn
from router import router_calihousing as calihousing
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calihousing.router)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)