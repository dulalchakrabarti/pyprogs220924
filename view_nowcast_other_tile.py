import folium
import json

# Load the GeoJSON file
with open("imd_nowcast.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# Map warning levels to colors
color_map = {
    "No Warning": "green",
    "Watch": "yellow",
    "Alert": "orange",
    "Warning": "red",
    "Unknown": "gray"
}

# Use CartoDB Positron tiles
m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")

# Add GeoJSON overlay
folium.GeoJson(
    geojson_data,
    style_function=lambda feature: {
        "fillColor": color_map.get(feature["properties"].get("WarningLevel"), "gray"),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.6,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["District", "State", "WarningLevel", "Date", "update_time"],
        aliases=["District:", "State:", "Warning:", "Date:", "Updated:"],
        localize=True
    )
).add_to(m)

# Add legend (custom HTML)
legend_html = """
<div style="
    position: fixed; 
    bottom: 30px; left: 30px; width: 150px; height: 140px; 
    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
    ">
    <b>Warning Levels</b><br>
    <i style="background:green; width:18px; height:18px; float:left; margin-right:8px;"></i>No Warning<br>
    <i style="background:yellow; width:18px; height:18px; float:left; margin-right:8px;"></i>Watch<br>
    <i style="background:orange; width:18px; height:18px; float:left; margin-right:8px;"></i>Alert<br>
    <i style="background:red; width:18px; height:18px; float:left; margin-right:8px;"></i>Warning<br>
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# Save map
m.save("imd_nowcast_map.html")

print("Interactive map with legend saved as imd_nowcast_map.html")
