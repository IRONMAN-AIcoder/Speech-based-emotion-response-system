import os
import urllib.request

port = os.environ.get("PORT", "8000")
urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
