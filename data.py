import requests
import json

x = requests.get("https://api.propublica.org/congress/v1/bills/search.json?query=infrastructure", headers = {'X-API-Key': '3qoRFu7imGjfAD2HoZQQuCcM3e37LbclqLe8VRv0'})
x_obj = json.loads(x.content) 
print(json.dumps(x_obj, indent=3))