import requests

url = "http://127.0.0.1:5000/parse"
data = {"value": "What is NN"}

response = requests.post(url, json=data,timeout=300)
xyz = response.json()

if response.status_code == 200:
    print("Response:", xyz['output']['result'])
else:
    print("Error:", response.status_code, response.text)
