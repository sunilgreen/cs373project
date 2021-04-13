import json

import requests

x = requests.get("https://api.propublica.org/congress/v1/112/bills/hr662.json", headers = {'X-API-Key': '3qoRFu7imGjfAD2HoZQQuCcM3e37LbclqLe8VRv0'})
x_obj = json.loads(x.content) 
#print(json.dumps(x_obj, indent=3))
o = x_obj["results"][0]
print(";".join([o['bill_id'], str(o["cosponsors"]), o["sponsor_party"], o["sponsor_state"], str(o["committee_codes"]), str(o["subcommittee_codes"]), o["primary_subject"]]))
