from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

class LocationData(BaseModel):
    uid: str
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    timestamp: datetime
    disasterType: Literal["non-disaster", "earthquake", "hurricane", "wildfire"]
