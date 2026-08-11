import requests
import time

start = time.time()
try:
    r = requests.post('http://localhost:8001/api/v1/auth/login', json={'username':'admin', 'password':'password'}, timeout=5)
    print(f"Status: {r.status_code}")
    print(f"Text: {r.text}")
except Exception as e:
    print(f"Error: {e}")
print(f"Time taken: {time.time() - start:.2f}s")
