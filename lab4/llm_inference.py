"""
实验四 阶段四：LLM 模型推理模块
调用本地 Ollama 模型，完成 Merge Prediction 和 Review Comment Generation 两个任务。
"""
import json
import os
import re
import time

import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

import config
from prompt_builder import build_prompt, PROMPT_TYPES

CONTEXT_TYPES = ["diff_only", "diff_pr_desc", "diff_commit", "diff_extra"]
TASKS = ["merge_prediction", "review_comment"]


def _get_result_path(task, prompt_type):
    """获取某一组实验的结果文件路径"""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    return os.path.join(config.RESULTS_DIR, f"{task}_{prompt_type}.json")


def call_llm(prompt):
    """
    根据 config.LLM_PROVIDER 自动选择底层 API 调用方式。
    返回 (response_text, latency_seconds)。
    """
    if config.LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    elif config.LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def _call_ollama(prompt):
    """调用本地 Ollama API，返回 (response_text, latency_seconds)。"""
    if len(prompt) > config.MAX_PROMPT_CHARS:
        prompt = prompt[:config.MAX_PROMPT_CHARS] + "\n\n... (truncated)"

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": config.OLLAMA_TEMPERATURE,
        "max_tokens": config.OLLAMA_MAX_TOKENS,
    }

    for attempt in range(config.MAX_RETRIES):
        try:
            start = time.time()
            resp = requests.post(
                config.OLLAMA_URL,
                json=payload,
                timeout=config.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            latency = time.time() - start
            text = resp.json().get("response", "")
            return text, latency

        except Exception as e:
            wait = 2 ** attempt
            print(f"  [Ollama 调用失败] {e}，{wait}s 后重试 ({attempt + 1}/{config.MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError("Ollama 调用失败，已达最大重试次数")


def _call_openai(prompt):
    """调用 OpenAI 兼容 API，返回 (response_text, latency_seconds)。"""
    if len(prompt) > config.MAX_PROMPT_CHARS:
        prompt = prompt[:config.MAX_PROMPT_CHARS] + "\n\n... (truncated)"

    payload = {
        "model": config.OPENAI_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": config.OPENAI_TEMPERATURE,
        "max_tokens": config.OPENAI_MAX_TOKENS,
    }
    if config.OPENAI_THINKING:
        payload["thinking"] = {"type": "enabled"}
    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(config.MAX_RETRIES):
        try:
            start = time.time()
            resp = requests.post(
                config.OPENAI_API_URL,
                json=payload,
                headers=headers,
                timeout=config.REQUEST_TIMEOUT,
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                else:
                    wait = 2 ** attempt
                raise Exception(f"429 Too Many Requests (Retry-After: {retry_after or 'N/A'})")

            resp.raise_for_status()
            latency = time.time() - start
            data = resp.json()

            if not data:
                raise ValueError("API 返回空响应体")

            choices = data.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                raise ValueError(f"API 返回异常 choices: {data}")

            message = choices[0].get("message")
            if not message:
                raise ValueError(f"API 返回异常 message: {data}")

            text = message.get("content")
            if text is None:
                finish_reason = choices[0].get("finish_reason", "N/A")
                print(f"  [警告] API 返回空内容，finish_reason={finish_reason}，完整响应: {data}")
                text = ""
            return text, latency

        except Exception as e:
            if "429" in str(e):
                pass
            else:
                wait = 2 ** attempt
            print(f"  [OpenAI 调用失败] {e}，{wait}s 后重试 ({attempt + 1}/{config.MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError("OpenAI 调用失败，已达最大重试次数")


def parse_merge_result(text):
    """
    从 LLM 回复中提取 JSON，解析出 decision 和 reason。
    只接受标准 JSON 格式；不接受正则降级匹配。
    如果 JSON 解析失败，返回 Unknown。
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

    return {"decision": "Unknown", "reason": "解析失败", "parse_error": True, "raw": text}


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
        key = (pr_id, task, ct, prompt_type)

        if key in completed:
            skipped += 1
            pbar.set_postfix({"new": len(results) - initial_count, "skip": skipped})
            continue

        context_text = ctx_entry["contexts"][ct]
        prompt = build_prompt(context_text, task, prompt_type)

        if task == "merge_prediction":
            total_latency = 0
            for attempt in range(config.JSON_RETRIES):
                response_text, latency = call_llm(prompt)
                total_latency += latency
                parsed = parse_merge_result(response_text)
                if not parsed["parse_error"]:
                    break
                if attempt < config.JSON_RETRIES - 1:
                    print(f"  [JSON 解析失败] PR {pr_id}({ct}) 第 {attempt+1} 次输出非 JSON，重试...")
            else:
                parsed = {
                    "decision": "Unknown",
                    "reason": f"{config.JSON_RETRIES}次重试仍无法解析为JSON",
                    "parse_error": True,
                    "raw": response_text,
                }
            latency = total_latency
        else:
            response_text, latency = call_llm(prompt)

        if task == "merge_prediction":
            record = {
                "pr_id": pr_id,
                "repo": repo,
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
            record = {
                "pr_id": pr_id,
                "repo": repo,
                "task": task,
                "context_type": ct,
                "prompt_type": prompt_type,
                "output_raw": response_text,
                "output_comment": response_text,
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

        time.sleep(config.REQUEST_INTERVAL)

    if len(results) != saved_at:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if skipped > 0:
        print(f"  跳过 {skipped} 条已有记录")
    if initial_count > 0:
        print(f"  加载已有 {initial_count} 条，新增 {len(results) - initial_count} 条")
    return results


def _compute_merge_metrics(results):
    """
    计算 Merge Prediction 的评估指标。
    需要 selected_prs.json 中的 label 作为 ground truth。
    """
    with open(config.SELECTED_PRS_PATH, "r", encoding="utf-8") as f:
        prs = json.load(f)
    label_map = {p["pr_id"]: p["label"] for p in prs}

    y_true = []
    y_pred = []
    y_score = []
    n_yes = 0
    n_no = 0
    n_unknown = 0

    for r in results:
        pr_id = r["pr_id"]
        if pr_id not in label_map:
            continue
        gt = label_map[pr_id]
        decision = r.get("output_decision", "Unknown")

        if decision == "Yes":
            pred = 1
            score = 1.0
            n_yes += 1
        elif decision == "No":
            pred = 0
            score = 0.0
            n_no += 1
        else:
            n_unknown += 1
            continue

        y_true.append(gt)
        y_pred.append(pred)
        y_score.append(score)

    if len(y_true) == 0:
        return None

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = None

    return {
        "n_yes": n_yes,
        "n_no": n_no,
        "n_unknown": n_unknown,
        "n_total": len(y_true),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def _print_group_metrics(results, task):
    """每组完成后打印统计指标"""
    if task != "merge_prediction":
        return

    m = _compute_merge_metrics(results)
    if m is None:
        return

    print(f"\n  预测分布: Yes={m['n_yes']}, No={m['n_no']}, Unknown={m['n_unknown']}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1-score : {m['f1']:.4f}")
    if m["roc_auc"] is not None:
        print(f"  ROC-AUC  : {m['roc_auc']:.4f}")
    else:
        print(f"  ROC-AUC  : N/A (只有单一类别)")


def run_all_experiments(contexts, force=False):
    """
    跑全部实验。每组保存到独立文件，每 10 条自动保存。

    文件命名: data/results/{task}_{prompt_type}.json

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
            _print_group_metrics(new_results, task)

    return all_results


def save_results(results, path=None):
    """保存全部实验结果到汇总文件"""
    if path is None:
        path = config.RESULTS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"汇总结果已保存到: {path}")
    print(f"文件大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
    print(f"总记录数: {len(results)}")


def print_result_summary(results):
    """打印各组实验的基本统计"""
    print("=" * 60)
    print("实验结果统计")
    print("=" * 60)

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
        print(f"{'Context':<16} {'Prompt':<14} {'Count':>6} {'AvgLatency':>10} {'ParseErr':>8}")
        print("-" * 56)
        for ct in CONTEXT_TYPES:
            for pt in PROMPT_TYPES:
                key = (task, ct, pt)
                if key in groups:
                    g = groups[key]
                    avg_lat = sum(g["latencies"]) / len(g["latencies"])
                    print(f"{ct:<16} {pt:<14} {g['total']:>6} {avg_lat:>9.2f}s {g['parse_errors']:>8}")