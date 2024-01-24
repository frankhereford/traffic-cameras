import datetime
import requests
from flask import jsonify


def status():
    # Get the current time
    current_time = datetime.datetime.now().isoformat()

    # Fetch weather data from NOAA API for Austin, TX (using station KAUS)
    response = requests.get("https://api.weather.gov/stations/KAUS/observations/latest")
    weather_data = response.json()

    # Get weather description and corresponding emoji
    weather_description = weather_data["properties"]["textDescription"]

    # Return the current time and brief weather description
    return jsonify(
        {
            "status": "nominal",
            "description": "traffic-cameras backend python processor",
            "current_time": current_time,
            "current_weather_in_austin": weather_description,
        }
    )
