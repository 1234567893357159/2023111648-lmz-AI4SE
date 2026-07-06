import requests
import certifi
print("cert证书路径：", certifi.where())
resp = requests.get("https://api.github.com/rate_limit", verify=certifi.where())
print("成功", resp.status_code)