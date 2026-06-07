import os
import uuid
import requests
from langchain.tools import tool
from typing import Dict
from dotenv import load_dotenv
import json
import cv2
import base64
import tempfile
from PIL import Image
import os
import uuid

load_dotenv()

api_key = os.getenv("WEATHER_API_KEY")

# Common function for WeatherAPI
def fetch_weather_data(location: str) -> Dict:
    url = "http://api.weatherapi.com/v1/current.json"
    params = {"key": api_key, "q": location, "aqi": "no"}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


@tool
def get_location_details(location: str) -> Dict:
    """
    Returns location details: name, region, country, latitude, longitude, timezone, and local time.
    """
    return fetch_weather_data(location).get("location", {})


@tool
def get_current_weather(location: str) -> Dict:
    """
    Returns current weather details including temperature, humidity, cloud cover, UV index, etc.
    """
    data = fetch_weather_data(location).get("current", {})
    return json.dumps(data)


@tool
def get_wind_details(location: str) -> Dict:
    """
    Returns wind details: wind speed, direction, and gust information.
    """
    current = fetch_weather_data(location).get("current", {})
    return {
        "wind_mph": current.get("wind_mph"),
        "wind_kph": current.get("wind_kph"),
        "wind_degree": current.get("wind_degree"),
        "wind_dir": current.get("wind_dir"),
        "gust_mph": current.get("gust_mph"),
        "gust_kph": current.get("gust_kph")
    }


basic_tools = [get_location_details, get_current_weather, get_wind_details]