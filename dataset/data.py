import json

import requests

# Pull non-pork bills from Propublica api in sequence
for i in range (1000,1200):
    x = requests.get("https://api.propublica.org/congress/v1/114/bills/hr" + str(i) + ".json", headers = {'X-API-Key': '3qoRFu7imGjfAD2HoZQQuCcM3e37LbclqLe8VRv0'})
    x_obj = json.loads(x.content) 

    o = x_obj["results"][0]
    
    if (len(o["committee_codes"]) > 0 and len(o["subcommittee_codes"]) > 0):
      print(";".join([o['bill_id'], str(o["cosponsors"]), o["sponsor_party"], o["sponsor_state"], str(o["committee_codes"]), str(o["subcommittee_codes"]), o["primary_subject"]]))

#Note: Bills were proofread after pulling and cross checked with Citizens Against Government waste to ensure that they were not pork barreled