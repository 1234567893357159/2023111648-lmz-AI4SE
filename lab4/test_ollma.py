import requests
import json

url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen2.5-coder:7b",
    "prompt": "请审查以下代码变更（diff）：\n\n@@ -10,7 +10,7 @@ def calculate(a, b):\n-    return a / b\n+    return a // b",
    "stream": False,   # 非流式，一次性返回
    "temperature": 0,
    "max_tokens": 512
}

response = requests.post(url, json=payload)
data = response.json()
print("审查意见：\n", data["response"])