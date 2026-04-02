import urllib.request
import urllib.error
import json
import time

# Create dummy expected columns based on error handling
req = urllib.request.Request(
    'http://127.0.0.1:8000/explain', 
    data=b'{"records":[{"sttl":1, "dbytes": 100}]}', 
    headers={'Content-Type': 'application/json'}
)
start = time.time()
try:
    response = urllib.request.urlopen(req, timeout=60)
    print('SUCCESS, HTTP 200, took', time.time() - start, 'seconds')
except urllib.error.HTTPError as e:
    print('HTTP ERROR:', e.code, e.read().decode())
except Exception as e:
    print('OTHER ERROR', e)
