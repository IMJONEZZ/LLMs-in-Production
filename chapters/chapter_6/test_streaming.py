import requests

url = "http://localhost:8000/generate"
data = """{"prompt": "Salt Lake City is a"}"""

with requests.post(url, data=data, stream=True) as r:
    for line in r.iter_lines(decode_unicode=True):
        print(line)
