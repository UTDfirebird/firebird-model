from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#import serverModelUse
import json

app = FastAPI()

class Tweet(BaseModel):
    key: str

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Tweet": "This concert is fire!"}

'''
# http://127.0.0.1:8000/items/5?q=somequery
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
'''
@app.post("/tweets/")
async def create_tweets(parsedData: dict):
    return parsedData
    #return serverModelUse.process_server_data(parsedData)
