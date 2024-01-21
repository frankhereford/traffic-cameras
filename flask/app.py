from flask import Flask, request, jsonify
import requests
import datetime
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)


@app.route("/")
def status():
    # Weather description to emoji mapping

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


@app.route("/vision", methods=["POST"])
def vision():
    logging.info("")
    data = request.get_json()
    base64_image = data.get("image")
    logging.info(f"Received image data: {base64_image[:100]}...")
    return jsonify({"status": "Image processed"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
