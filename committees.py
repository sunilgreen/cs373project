import json

import requests

# Pull the committee information from propublica

x = requests.get("https://api.propublica.org/congress/v1/115/senate/committees.json", headers = {'X-API-Key': '3qoRFu7imGjfAD2HoZQQuCcM3e37LbclqLe8VRv0'})
x_obj = json.loads(x.content) 
print(json.dumps(x_obj, indent=3))
