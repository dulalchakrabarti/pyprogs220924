import requests
import json
import pandas as pd

url = "https://reactjs.imd.gov.in/geoserver/imd/wfs?callback=getJson&service=WFS&version=1.1.0&request=GetFeature&typename=imd%3ANowcastWarningDistrict&srsname=EPSG%3A4326&format_options=callback%3AgetJson&outputFormat=text%2Fjavascript"

resp = requests.get(url)
text = resp.text

# Strip JSONP wrapper
clean = text[text.find("(")+1 : text.rfind(")")]

data = json.loads(clean)
features = data.get("features", [])

# Map Color codes to warning levels
color_map = {
    0: "No Warning",
    1: "Watch",
    2: "Alert",
    3: "Warning"
}

rows = []
for f in features:
    props = f.get("properties", {})
    district = props.get("District")
    color = props.get("Color")
    warning = color_map.get(color, "Unknown")
    if district:
        rows.append({
            "District": district,
            "Warning": warning,
            "State": props.get("State"),
            "Date": props.get("Date"),
            "UpdateTime": props.get("update_time")
        })

# Save CSV
df = pd.DataFrame(rows)
df.to_csv("imd_nowcast.csv", index=False, encoding="utf-8")

# Save GeoJSON (keep full geometry + properties)
geojson = {
    "type": "FeatureCollection",
    "features": []
}
for f in features:
    props = f.get("properties", {})
    district = props.get("District")
    color = props.get("Color")
    warning = color_map.get(color, "Unknown")
    if district:
        # enrich properties with readable warning
        props["WarningLevel"] = warning
        geojson["features"].append({
            "type": "Feature",
            "geometry": f.get("geometry"),
            "properties": props
        })

with open("imd_nowcast.geojson", "w", encoding="utf-8") as g:
    json.dump(geojson, g, ensure_ascii=False, indent=2)

print("Saved", len(df), "district warnings to imd_nowcast.csv and imd_nowcast.geojson")
