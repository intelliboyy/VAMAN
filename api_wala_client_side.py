import requests

res = requests.post("http://localhost:8000/ask", json={"question": "Who is the director of PSIT?"})
data = res.json()

print(data.get("response_text", data))
