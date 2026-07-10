"""
实验六 步骤四：LLM 模型推理模块
调用本地 Ollama 模型，完成 Merge Prediction 和 Review Comment Generation 两个任务。

实验矩阵: 5 上下文 × 4 Prompt × 2 任务 = 40 组实验
每组结果保存到独立 JSON 文件，支持断点续传。
"""

import json
import os
import re
import time

import requests
from tqdm import tqdm

import config
from prompt_builder import build_prompt

CONTEXT_TYPES = config.CONTEXT_TYPES
PROMPT_TYPES = config.PROMPT_TYPES
TASKS = config.TASK_TYPES

STEP4_RESULTS_DIR = os.path.join(config.LAB6_RESULTS_DIR, "step4")
os.makedirs(STEP4_RESULTS_DIR, exist_ok=True)


def _get_result_path(task, prompt_type):
    """获取某一组实验的结果文件路径"""
    return os.path.join(STEP4_RESULTS_DIR, f"{task}_{prompt_type}.json")


def call_llm(prompt):
    """
    调用本地 Ollama 模型，返回 (response_text, latency_seconds)。

    失败时自动重试，最多 config.LLM_MAX_RETRIES 次。
    """
    if len(prompt) > config.LLM_MAX_PROMPT_CHARS:
        prompt = prompt[:config.LLM_MAX_PROMPT_CHARS] + "\n\n... (truncated)"

    payload = {
        "model": config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
    }

    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            start = time.time()
            resp = requests.post(
                config.LLM_API_URL,
                json=payload,
                timeout=config.LLM_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            latency = time.time() - start
            text = resp.json().get("response", "")
            return text, latency

        except Exception as e:
            wait = 2 ** attempt
            print(f"  [Ollama 调用失败] {e}，{wait}s 后重试 ({attempt + 1}/{config.LLM_MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError("Ollama 调用失败，已达最大重试次数")


def parse_merge_result(text):
    """
    从 LLM 回复中提取 JSON，解析出 decision 和 reason。

    1. 优先匹配 JSON 块并解析
    2. JSON 解析失败则降级匹配 Yes/No 关键词
    3. 都失败则标记为 Unknown
    """
    if not text:
        return {"decision": "Unknown", "reason": "空响应", "parse_error": True, "raw": ""}

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            decision = obj.get("decision", "Unknown")
            reason = obj.get("reason", "")

            if decision in ("Yes", "No"):
                return {
                    "decision": decision,
                    "reason": reason,
                    "parse_error": False,
                    "raw": text,
                }

            if decision != "Unknown":
                return {
                    "decision": decision,
                    "reason": reason,
                    "parse_error": True,
                    "raw": text,
                }

        except json.JSONDecodeError:
            pass

    yes_match = re.search(r"\b(YES|Yes|yes)\b", text)
    if yes_match:
        return {"decision": "Yes", "reason": "从原文提取", "parse_error": True, "raw": text}
    no_match = re.search(r"\b(NO|No|no)\b", text)
    if no_match:
        return {"decision": "No", "reason": "从原文提取", "parse_error": True, "raw": text}

    return {"decision": "Unknown", "reason": "解析失败", "parse_error": True, "raw": text}


def parse_self_reflection_merge(text):
    """
    专门解析 Self-Reflection 的输出：从 [FINAL JUDGMENT] 块中提取 JSON。

    三级降级策略：
    1. 匹配 [FINAL JUDGMENT] 后的 JSON
    2. 取最后一个 JSON 块
    3. 降级到通用 parse_merge_result
    """
    if not text:
        return parse_merge_result(text)

    final_match = re.search(
        r'\[FINAL JUDGMENT\]\s*\n?\s*(\{[^{}]*\})',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if final_match:
        try:
            obj = json.loads(final_match.group(1))
            decision = obj.get("decision", "Unknown")
            reason = obj.get("reason", "")
            if decision in ("Yes", "No"):
                return {
                    "decision": decision,
                    "reason": reason,
                    "parse_error": False,
                    "raw": text,
                }
        except json.JSONDecodeError:
            pass

    all_json = re.findall(r'\{[^{}]*\}', text)
    if all_json:
        try:
            obj = json.loads(all_json[-1])
            decision = obj.get("decision", "Unknown")
            if decision in ("Yes", "No"):
                return {
                    "decision": decision,
                    "reason": obj.get("reason", ""),
                    "parse_error": False,
                    "raw": text,
                }
        except json.JSONDecodeError:
            pass

    return parse_merge_result(text)


def parse_self_reflection_review(text):
    """
    从 Self-Reflection 输出中提取 [FINAL REVIEW] 部分的内容。

    三级降级：
    1. 匹配 [FINAL REVIEW] 后的内容
    2. 取 [INITIAL REVIEW] 和 [FINAL REVIEW] 之间的内容
    3. 直接返回原始文本
    """
    if not text:
        return text

    final_match = re.search(
        r'\[FINAL REVIEW\]\s*\n(.*?)$',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if final_match:
        final_text = final_match.group(1).strip()
        if final_text:
            return final_text

    return text.strip()


def load_all_results():
    """加载所有组的结果文件，合并返回"""
    all_results = []
    for task in TASKS:
        for pt in PROMPT_TYPES:
            path = _get_result_path(task, pt)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))
    if all_results:
        print(f"已加载已有结果: {len(all_results)} 条记录")
    return all_results


def _build_completed_set(results):
    """从已有结果中构建已完成记录的集合"""
    return {
        (r["pr_id"], r["task"], r["context_type"], r["prompt_type"])
        for r in results
    }


def run_single_experiment(contexts, task, prompt_type, completed=None):
    """
    对全部 PR 跑一组实验，跳过已完成的记录。
    每 10 条结果保存一次。

    返回:
        list[dict]: 本组全部结果（含从文件加载的旧结果）
    """
    if completed is None:
        completed = set()

    result_path = _get_result_path(task, prompt_type)

    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    initial_count = len(results)

    items = []
    for ctx_entry in contexts:
        for ct in CONTEXT_TYPES:
            items.append((ctx_entry, ct))

    skipped = 0
    saved_at = len(results)
    pbar = tqdm(items, desc=f"[{task}][{prompt_type}]", unit="req", ncols=100)

    for ctx_entry, ct in pbar:
        pr_id = ctx_entry["pr_id"]
        repo = ctx_entry["repo"]
        label = ctx_entry.get("label", 0)
        key = (pr_id, task, ct, prompt_type)

        if key in completed:
            skipped += 1
            pbar.set_postfix({"new": len(results) - initial_count, "skip": skipped})
            continue

        context_text = ctx_entry["contexts"].get(ct, "")
        if not context_text:
            continue

        prompt = build_prompt(context_text, task, prompt_type)
        response_text, latency = call_llm(prompt)

        if task == "merge_prediction":
            if prompt_type == "self_reflection":
                parsed = parse_self_reflection_merge(response_text)
            else:
                parsed = parse_merge_result(response_text)
            record = {
                "pr_id": pr_id,
                "repo": repo,
                "label": label,
                "task": task,
                "context_type": ct,
                "prompt_type": prompt_type,
                "output_raw": response_text,
                "output_decision": parsed["decision"],
                "output_reason": parsed["reason"],
                "parse_error": parsed["parse_error"],
                "latency": round(latency, 2),
            }
        else:
            if prompt_type == "self_reflection":
                comment = parse_self_reflection_review(response_text)
            else:
                comment = response_text
            record = {
                "pr_id": pr_id,
                "repo": repo,
                "label": label,
                "task": task,
                "context_type": ct,
                "prompt_type": prompt_type,
                "output_raw": response_text,
                "output_comment": comment,
                "latency": round(latency, 2),
            }

        results.append(record)
        pbar.set_postfix({
            "new": len(results) - initial_count,
            "lat": f"{latency:.1f}s",
            "skip": skipped,
        })

        if len(results) % 10 == 0 and len(results) != saved_at:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            saved_at = len(results)

        time.sleep(config.LLM_REQUEST_INTERVAL)

    if len(results) != saved_at:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if skipped > 0:
        print(f"  跳过 {skipped} 条已有记录")
    if initial_count > 0:
        print(f"  加载已有 {initial_count} 条，新增 {len(results) - initial_count} 条")
    return results


def run_all_experiments(contexts, force=False):
    """
    跑全部 40 组实验。每组保存到独立文件，每 10 条自动保存。

    文件命名: results/step4/{task}_{prompt_type}.json

    参数:
        force: 为 True 时强制重新跑全部实验
    """
    total_groups = len(TASKS) * len(PROMPT_TYPES)

    if force:
        completed = set()
        all_results = []
    else:
        all_results = load_all_results()
        completed = _build_completed_set(all_results)

    group = 0
    for task in TASKS:
        for pt in PROMPT_TYPES:
            group += 1
            print(f"\n{'=' * 60}")
            print(f"组 {group}/{total_groups}: {task} + {pt}")
            print(f"  文件: {_get_result_path(task, pt)}")
            print(f"{'=' * 60}")
            new_results = run_single_experiment(contexts, task, pt, completed)
            all_results.extend(new_results)
            completed.update(
                (r["pr_id"], r["task"], r["context_type"], r["prompt_type"])
                for r in new_results
            )
            print(f"  本组完成: {len(new_results)} 条记录")

    return all_results


def save_results(results, path=None):
    """保存全部实验结果到汇总文件"""
    if path is None:
        path = os.path.join(STEP4_RESULTS_DIR, "all_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"汇总结果已保存到: {path}")
    print(f"文件大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
    print(f"总记录数: {len(results)}")


def print_result_summary(results):
    """打印各组实验的基本统计"""
    print("=" * 70)
    print("实验结果统计")
    print("=" * 70)

    groups = {}
    for r in results:
        key = (r["task"], r["context_type"], r["prompt_type"])
        if key not in groups:
            groups[key] = {"total": 0, "latencies": [], "parse_errors": 0}
        g = groups[key]
        g["total"] += 1
        g["latencies"].append(r["latency"])
        if r.get("parse_error"):
            g["parse_errors"] += 1

    for task in TASKS:
        print(f"\n--- {task} ---")
        header = f"{'Context':<16} {'Prompt':<18} {'Count':>6} {'AvgLat':>8} {'ParseErr':>8}"
        print(header)
        print("-" * 60)
        for ct in CONTEXT_TYPES:
            for pt in PROMPT_TYPES:
                key = (task, ct, pt)
                if key in groups:
                    g = groups[key]
                    avg_lat = sum(g["latencies"]) / len(g["latencies"])
                    print(f"{ct:<16} {pt:<18} {g['total']:>6} {avg_lat:>7.1f}s {g['parse_errors']:>8}")


if __name__ == "__main__":
    import json as _json
    from prompt_builder import build_prompt

    with open(config.CONTEXTS_PATH, "r", encoding="utf-8") as _f:
        _contexts = _json.load(_f)

    _results = run_all_experiments(_contexts)
    save_results(_results)
    print_result_summary(_results)