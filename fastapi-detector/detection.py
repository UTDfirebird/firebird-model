from models import LocationData
from datetime import datetime
from typing import List, Tuple
import uuid

# list to store recent locations
recentLocations = []

# list to store active disasters
activeDisasters = []

# adjust parameters based on testing
detectionNumber = 5
clearTime = 10  # minutes
locRadius = 0.43  # ~30 miles

class ActiveDisaster:
    def __init__(self, disasterType: str, initialLocations: List[LocationData]):
        self.disasterId = str(uuid.uuid4())  # unique ID
        self.disasterType = disasterType
        self.locations = initialLocations[:]
        self.tweetCount = len(initialLocations)
        self.detectionTime = min(loc.timestamp for loc in initialLocations)

        # calculate bounding box
        lats = [loc.latitude for loc in initialLocations]
        longs = [loc.longitude for loc in initialLocations]
        self.boundingBox = (min(lats), max(lats), min(longs), max(longs))

        # calculate centroid
        avgLat = sum(lats) / len(lats)
        avgLong = sum(longs) / len(longs)
        self.centroid = (avgLat, avgLong)

    def addLocations(self, newLocations: List[LocationData]):
        self.locations.extend(newLocations)
        self.tweetCount += len(newLocations)

        allLats = [loc.latitude for loc in self.locations]
        allLongs = [loc.longitude for loc in self.locations]
        self.boundingBox = (min(allLats), max(allLats), min(allLongs), max(allLongs))
        self.centroid = (
            sum(allLats) / len(allLats),
            sum(allLongs) / len(allLongs)
        )

def detect_disaster(data: LocationData) -> dict:
    global recentLocations, activeDisasters

    now = datetime.utcnow()

    # remove outdated tweets unless they're part of active disasters
    recentLocations = [
        loc for loc in recentLocations
        if any(loc in disaster.locations for disaster in activeDisasters) or
        (now - loc.timestamp).total_seconds() <= clearTime * 60
    ]

    if data.disasterType == "non-disaster":
        return {
            "disaster_detected": False,
            "tweetCount": 0
        }

    # add new tweet
    recentLocations.append(data)

    # find tweets of the same type nearby
    tweetMatch = [
        loc for loc in recentLocations
        if loc.disasterType == data.disasterType and
           abs(loc.latitude - data.latitude) <= locRadius and
           abs(loc.longitude - data.longitude) <= locRadius
    ]

    isDisaster = len(tweetMatch) >= detectionNumber

    if isDisaster:
        matchedDisaster = None
        for active in activeDisasters:
            if active.disasterType == data.disasterType:
                minLat, maxLat, minLong, maxLong = active.boundingBox
                if (minLat <= data.latitude <= maxLat and minLong <= data.longitude <= maxLong):
                    matchedDisaster = active
                    break

        if matchedDisaster:
            matchedDisaster.addLocations(tweetMatch)
            return {
                "disaster_detected": True,
                "disasterId": matchedDisaster.disasterId,
                "tweetCount": matchedDisaster.tweetCount,
                "boundingBox": matchedDisaster.boundingBox,
                "centroid": matchedDisaster.centroid,
            }
        else:
            newDisaster = ActiveDisaster(data.disasterType, tweetMatch)
            activeDisasters.append(newDisaster)
            return {
                "disaster_detected": True,
                "disasterId": newDisaster.disasterId,
                "tweetCount": newDisaster.tweetCount,
                "boundingBox": newDisaster.boundingBox,
                "centroid": newDisaster.centroid,
            }

    return {
        "disaster_detected": False,
        "tweetCount": len(tweetMatch)
    }
