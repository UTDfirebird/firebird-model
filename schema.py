from pydantic import BaseModel, Field
from typing import List, Optional
from decimal import Decimal

class User(BaseModel):
    username: str
    verified: bool
    followers_count: int

class OriginalTweetData(BaseModel):
    tweet_text: str
    timestamp: str
    user: User
    hashtags: List[str]

class Details(BaseModel):
    urgency: Decimal = Field(ge=0, le=1)
    emotional_impact: Decimal = Field(ge=0, le=1)

class Sentiment(BaseModel):
    emotion: str
    tone: str
    polarity: Decimal = Field(ge=-1, le=1)
    details: Details

class AnalysisItem(BaseModel):
    disaster_name: str
    relevant_to_disaster: Decimal = Field(ge=0, le=1)
    sentiment: Sentiment
    sarcasm_confidence: Decimal = Field(ge=0, le=1)

class TweetItem(BaseModel):
    original_tweet_data: OriginalTweetData
    analysis: List[AnalysisItem]

class Tweets(BaseModel):
    tweets: List[TweetItem]