"""
ModelScope API 配额查询工具
查看用户当天和模型当天的请求限额与剩余额度。

用法:
    python check_modelscope_quota.py
"""

import sys
import os
import time
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lab4"))
import config


def check_quota():
    """发送一个最小请求，从响应头中提取配额信息。"""

    payload = {
        "model": config.OPENAI_MODEL,
        "messages": [
            {"role": "user", "content": "Hi"}
        ],
        "stream": False,
        "temperature": config.OPENAI_TEMPERATURE,
        "max_tokens": 1,
    }
    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    print("=" * 60)
    print("  ModelScope API 配额查询")
    print("=" * 60)
    print(f"  API URL : {config.OPENAI_API_URL}")
    print(f"  Model   : {config.OPENAI_MODEL}")
    print()

    try:
        start = time.time()
        resp = requests.post(
            config.OPENAI_API_URL,
            json=payload,
            headers=headers,
            timeout=config.REQUEST_TIMEOUT,
        )
        latency = time.time() - start
        resp.raise_for_status()

        print(f"  [状态码] {resp.status_code}  (耗时 {latency:.2f}s)")
        print()

        print("-" * 60)
        print("  响应头中的配额信息 (Rate Limit Headers)")
        print("-" * 60)

        rate_limit_headers = {
            "modelscope-ratelimit-requests-limit": "用户当天限额",
            "modelscope-ratelimit-requests-remaining": "用户当天剩余额度",
            "modelscope-ratelimit-model-requests-limit": "模型当天限额",
            "modelscope-ratelimit-model-requests-remaining": "模型当天剩余额度",
        }

        found_any = False
        for header_key, description in rate_limit_headers.items():
            value = resp.headers.get(header_key, "N/A")
            if value != "N/A":
                found_any = True
            print(f"  {description:20s} : {value}")

        if not found_any:
            print()
            print("  [提示] 未在响应头中找到配额信息。")
            print("  可能原因：API 暂未返回这些头部，或头部名称有变化。")
            print()
            print("  所有响应头如下：")
            print("-" * 60)
            for key, value in resp.headers.items():
                print(f"  {key}: {value}")

        print()
        print("-" * 60)

        remaining = resp.headers.get("modelscope-ratelimit-requests-remaining")
        if remaining and remaining.isdigit():
            remaining = int(remaining)
            if remaining < 100:
                print(f"  ⚠ 警告：用户剩余额度仅 {remaining}，建议注意用量！")
            else:
                print(f"  ✓ 剩余额度充足 ({remaining})")

        model_remaining = resp.headers.get("modelscope-ratelimit-model-requests-remaining")
        if model_remaining and model_remaining.isdigit():
            model_remaining = int(model_remaining)
            if model_remaining < 50:
                print(f"  ⚠ 警告：模型剩余额度仅 {model_remaining}，建议注意用量！")
            else:
                print(f"  ✓ 模型剩余额度充足 ({model_remaining})")

        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("  [错误] 无法连接到 API 服务器，请检查网络。")
    except requests.exceptions.Timeout:
        print(f"  [错误] 请求超时 (>{config.REQUEST_TIMEOUT}s)，请稍后重试。")
    except requests.exceptions.HTTPError as e:
        print(f"  [HTTP 错误] {e}")
        if resp is not None:
            print(f"  响应体: {resp.text[:500]}")
    except Exception as e:
        print(f"  [错误] {e}")


if __name__ == "__main__":
    check_quota()