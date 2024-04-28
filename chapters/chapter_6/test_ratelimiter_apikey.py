# First start server
# python chapters/chapter_6/listing_6.2.flow_control.py

import requests
import time

# Url for the API endpoint
url = "http://localhost:8000/hello"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 1234567abcdefg",
}

bad_headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer password",
}

# Make a request with the correct API key
response = requests.get(url, headers=headers)
assert response.status_code == 200
print(response.json())

# Make a request with the wrong API key
response = requests.get(url, headers=bad_headers)
assert response.status_code == 401
print(response.json())

# Make multiple requests to test the rate limiter
for i in range(6):
    response = requests.get(url, headers=headers)
    print(
        f"Request {i+1}: Status code: {response.status_code}, \
          Response: {response.text}"
    )
    time.sleep(
        1
    )  # Sleep for 1 second to make the requests in a spaced out manner
