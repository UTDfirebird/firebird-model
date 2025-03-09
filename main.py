from typing import Union
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from schema import Tweets
from pydantic import ValidationError
#import serverModelUse
import json

app = FastAPI()

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
@app.get("/tweets/", include_in_schema=False)
async def redirect_tweets(request: Request):
    scheme = request.headers.get("X-Forwarded-Proto", "https")
    new_location = f"{scheme}://{request.headers.get('Host')}/tweets"
    return RedirectResponse(url=new_location, status_code=307)

@app.post("/tweets")
async def create_tweets(parsedData: dict, request: Request):
    print(f"X-Forwarded-Proto: {request.headers.get('X-Forwarded-Proto')}")
    return parsedData

@app.post("/tweets/")
async def create_tweets_with_slash(parsedData: dict, request: Request):
    return await create_tweets(parsedData)
