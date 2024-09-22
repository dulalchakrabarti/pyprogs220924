import openmeteo_requests
from openmeteo_sdk.Variable import Variable

om = openmeteo_requests.Client()
params = {
    "latitude": 52.54,
    "longitude": 13.41,
    "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
    "current": ["temperature_2m", "relative_humidity_2m"]
}

responses = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Current values
current = response.Current()
current_variables = list(map(lambda i: current.Variables(i), range(0, current.VariablesLength())))
current_temperature_2m = next(filter(lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2, current_variables))
current_relative_humidity_2m = next(filter(lambda x: x.Variable() == Variable.relative_humidity and x.Altitude() == 2, current_variables))

print(f"Current time {current.Time()}")
print(f"Current temperature_2m {current_temperature_2m.Value()}")
print(f"Current relative_humidity_2m {current_relative_humidity_2m.Value()}")
'''
import numpy as np

hourly = response.Hourly()
hourly_time = range(hourly.Time(), hourly.TimeEnd(), hourly.Interval())
hourly_variables = list(map(lambda i: hourly.Variables(i), range(0, hourly.VariablesLength())))

hourly_temperature_2m = next(filter(lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2, hourly_variables)).ValuesAsNumpy()
hourly_precipitation = next(filter(lambda x: x.Variable() == Variable.precipitation, hourly_variables)).ValuesAsNumpy()
hourly_wind_speed_10m = next(filter(lambda x: x.Variable() == Variable.wind_speed and x.Altitude() == 10, hourly_variables)).ValuesAsNumpy()


import pandas as pd

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s"),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

hourly_dataframe_pd = pd.DataFrame(data = hourly_data)
print(hourly_dataframe_pd)

import polars as pl
from datetime import datetime, timedelta, timezone

start = datetime.fromtimestamp(hourly.Time(), timezone.utc)
end = datetime.fromtimestamp(hourly.TimeEnd(), timezone.utc)
freq = timedelta(seconds = hourly.Interval())

hourly_dataframe_pl = pl.select(
    date = pl.datetime_range(start, end, freq, closed = "left"),
    temperature_2m = hourly_temperature_2m,
    precipitation = hourly_precipitation,
    wind_speed_10m = hourly_wind_speed_10m
)
print(hourly_dataframe_pl)
'''