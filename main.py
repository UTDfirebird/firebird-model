from typing import Union
from fastapi import FastAPI
from schema import Tweets
from pydantic import ValidationError

app = FastAPI()


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
async def create_tweets(tweets: Tweets):
    try:
        # TODO: Process the tweet data with the model (not ready yet)
        
        return tweets
    except ValidationError as e:
        return e.json()
