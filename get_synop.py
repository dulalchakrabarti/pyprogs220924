import requests
import json
url = 'https://wis2node.wis.cma.cn/oapi/collections/urn:wmo:md:cn-cma:data.core.weather.surface-based-observations/items?f=json&datetime=2024-10-25&index=17&wigos_station_identifier=0-20000-0-51334'
r = requests.get(url)
data_json = r.json()
print(json.dumps(data_json, indent=4, sort_keys=True))

