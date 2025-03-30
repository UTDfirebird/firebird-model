from fastapi import FastAPI
from models import LocationData
from detection import detect_disaster

app = FastAPI()

@app.post("/location")
async def receive_location(data: LocationData):
    result = detect_disaster(data)
    print(f"Received location data: {data}")
    print(f"Detection result: {result}")
    return {
        "status": "success",
        "received": data.dict(),
        "detection_result": result
    }
