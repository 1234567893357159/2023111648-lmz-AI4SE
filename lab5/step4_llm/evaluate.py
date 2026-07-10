"""
步骤四：LLM 大模型评估
使用 Ollama 对 AI 生成代码进行代码审查
- Task 1: Merge Prediction（是否合并）
- Task 2: Review Comment Generation（代码审查评论）
- 4 种 Prompt: zero_shot / few_shot / cot / role_based
- 4 种 Context: diff_only / diff_pr_desc / diff_commit / diff_extra

断点续传：结果按 (task, prompt_type) 保存到独立文件，中断后重跑自动跳过已完成记录。
每 10 条保存一次。
"""

import json
import math
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "lab4"))

import config
from prompt_builder import build_prompt, PROMPT_TYPES

CONTEXT_TYPES = ["diff_only", "diff_pr_desc", "diff_commit", "diff_extra"]
TASKS = ["merge_prediction", "review_comment"]

OLLAMA_URL = config.OLLAMA_URL
OLLAMA_MODEL = config.OLLAMA_MODEL
OLLAMA_TEMPERATURE = config.OLLAMA_TEMPERATURE
OLLAMA_MAX_TOKENS = config.OLLAMA_MAX_TOKENS
MAX_RETRIES = config.MAX_RETRIES
REQUEST_TIMEOUT = config.REQUEST_TIMEOUT
MAX_PROMPT_CHARS = config.MAX_PROMPT_CHARS

STEP4_RESULTS_DIR = os.path.join(config.LAB5_RESULTS_DIR, "step4")
STEP4_CONTEXTS_PATH = os.path.join(config.LAB5_DATA_DIR, "step4_contexts.json")
STEP4_SUMMARY_PATH = os.path.join(STEP4_RESULTS_DIR, "summary.json")

os.makedirs(STEP4_RESULTS_DIR, exist_ok=True)


def _get_result_path(task, prompt_type):
    """获取某一组实验的结果文件路径"""
    return os.path.join(STEP4_RESULTS_DIR, f"{task}_{prompt_type}.json")


def call_llm(prompt):
    """调用本地 Ollama 模型，返回 (response_text, latency)"""
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS] + "\n\n... (truncated)"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": OLLAMA_TEMPERATURE,
        "max_tokens": OLLAMA_MAX_TOKENS,
    }

    for attempt in range(MAX_RETRIES):
        try:
            start = time.time()
            resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            latency = time.time() - start
            text = resp.json().get("response", "")
            return text, latency
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [Ollama 调用失败] {e}，{wait}s 后重试 ({attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError("Ollama 调用失败，已达最大重试次数")


def parse_merge_result(text):
    """从 LLM 回复中提取 JSON，解析出 decision 和 reason"""
    if not text:
        return {"decision": "Unknown", "reason": "空响应", "parse_error": True, "raw": ""}

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            decision = obj.get("decision", "Unknown")
            reason = obj.get("reason", "")
            if decision in ("Yes", "No"):
                return {"decision": decision, "reason": reason, "parse_error": False, "raw": text}
            if decision != "Unknown":
                return {"decision": decision, "reason": reason, "parse_error": True, "raw": text}
        except json.JSONDecodeError:
            pass

    yes_match = re.search(r"\b(YES|Yes|yes)\b", text)
    if yes_match:
        return {"decision": "Yes", "reason": "从原文提取", "parse_error": True, "raw": text}
    no_match = re.search(r"\b(NO|No|no)\b", text)
    if no_match:
        return {"decision": "No", "reason": "从原文提取", "parse_error": True, "raw": text}

    return {"decision": "Unknown", "reason": "解析失败", "parse_error": True, "raw": text}


def _tokenize(text):
    return text.lower().split()


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(reference_tokens, candidate_tokens, max_n=4):
    if not candidate_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(_ngrams(reference_tokens, n))
        cand_ngrams = Counter(_ngrams(candidate_tokens, n))
        match_count = sum((ref_ngrams & cand_ngrams).values())
        total = max(len(candidate_tokens) - n + 1, 1)
        precisions.append(match_count / total if total > 0 else 0.0)
    if all(p == 0.0 for p in precisions):
        return 0.0
    bp = 1.0 if len(candidate_tokens) >= len(reference_tokens) else \
        math.exp(1 - len(reference_tokens) / max(len(candidate_tokens), 1))
    log_avg = sum(math.log(p) for p in precisions if p > 0) / len(precisions)
    return bp * math.exp(log_avg)


def compute_rouge_l(reference_tokens, candidate_tokens):
    m, n = len(reference_tokens), len(candidate_tokens)
    if m == 0 or n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference_tokens[i - 1] == candidate_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


class LLMEvaluator:
    """LLM 评估器：使用 Ollama 对 AI 生成代码进行代码审查，支持断点续传"""

    def __init__(self):
        self.contexts = []
        self.all_results = []
        self.completed = set()
        self.summary = {}

    def load_contexts(self):
        if os.path.exists(STEP4_CONTEXTS_PATH):
            print(f"加载上下文缓存: {STEP4_CONTEXTS_PATH}")
            with open(STEP4_CONTEXTS_PATH, "r", encoding="utf-8") as f:
                self.contexts = json.load(f)
            print(f"  已加载 {len(self.contexts)} 条上下文")
        else:
            from data_prepare import AIPRDataPreparer
            preparer = AIPRDataPreparer()
            self.contexts, _ = preparer.run_all()
            with open(STEP4_CONTEXTS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.contexts, f, indent=2, ensure_ascii=False)
            print(f"  上下文已保存: {STEP4_CONTEXTS_PATH}")
        return self.contexts

    def load_all_results(self):
        """加载所有已有结果，构建已完成集合"""
        self.all_results = []
        for task in TASKS:
            for pt in PROMPT_TYPES:
                path = _get_result_path(task, pt)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        self.all_results.extend(json.load(f))
        if self.all_results:
            print(f"已加载已有结果: {len(self.all_results)} 条记录")
        self.completed = self._build_completed_set()
        return self.all_results

    def _build_completed_set(self):
        """从已有结果中构建已完成记录的集合"""
        return {
            (r["pr_id"], r["task"], r["context_type"], r["prompt_type"])
            for r in self.all_results
        }

    def run_single_experiment(self, task, prompt_type):
        """对全部 PR 跑一组实验，跳过已完成的记录，每 10 条保存"""
        result_path = _get_result_path(task, prompt_type)

        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            results = []

        initial_count = len(results)

        items = []
        for ctx_entry in self.contexts:
            for ct in CONTEXT_TYPES:
                items.append((ctx_entry, ct))

        skipped = 0
        saved_at = len(results)
        pbar = tqdm(items, desc=f"[{task}][{prompt_type}]", unit="req", ncols=100)

        for ctx_entry, ct in pbar:
            pr_id = ctx_entry["pr_id"]
            repo = ctx_entry["repo"]
            key = (pr_id, task, ct, prompt_type)

            if key in self.completed:
                skipped += 1
                pbar.set_postfix({"new": len(results) - initial_count, "skip": skipped})
                continue

            context_text = ctx_entry["contexts"].get(ct, "")
            if not context_text:
                continue

            prompt = build_prompt(context_text, task, prompt_type)
            response_text, latency = call_llm(prompt)

            if task == "merge_prediction":
                parsed = parse_merge_result(response_text)
                label = ctx_entry.get("label", 0)
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

        if len(results) != saved_at:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        if skipped > 0:
            print(f"  跳过 {skipped} 条已有记录")
        if initial_count > 0:
            print(f"  加载已有 {initial_count} 条，新增 {len(results) - initial_count} 条")
        return results

    def run_all_experiments(self):
        """跑全部 8 组实验（2 任务 × 4 Prompt），支持断点续传"""
        total_groups = len(TASKS) * len(PROMPT_TYPES)

        self.load_all_results()

        group = 0
        for task in TASKS:
            for pt in PROMPT_TYPES:
                group += 1
                print(f"\n{'=' * 60}")
                print(f"组 {group}/{total_groups}: {task} + {pt}")
                print(f"  文件: {_get_result_path(task, pt)}")
                print(f"{'=' * 60}")
                new_results = self.run_single_experiment(task, pt)
                self.all_results.extend(new_results)
                self.completed.update(
                    (r["pr_id"], r["task"], r["context_type"], r["prompt_type"])
                    for r in new_results
                )
                print(f"  本组完成: {len(new_results)} 条记录")
                self._print_group_metrics(new_results, task)

        return self.all_results

    def _print_group_metrics(self, results, task):
        """每组完成后打印统计指标"""
        if task != "merge_prediction":
            return
        m = self._compute_merge_metrics(results)
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

    def _compute_merge_metrics(self, results):
        """计算 Merge Prediction 评估指标"""
        y_true = []
        y_pred = []
        y_score = []
        n_yes = n_no = n_unknown = 0

        for r in results:
            label = r.get("label", 0)
            decision = r.get("output_decision", "Unknown")

            if decision == "Yes":
                pred, score = 1, 1.0
                n_yes += 1
            elif decision == "No":
                pred, score = 0, 0.0
                n_no += 1
            else:
                pred, score = 0, 0.5
                n_unknown += 1

            y_true.append(label)
            y_pred.append(pred)
            y_score.append(score)

        if len(y_true) == 0:
            return None

        if len(set(y_true)) < 2:
            return {
                "n_samples": len(y_true), "n_yes": n_yes,
                "n_no": n_no, "n_unknown": n_unknown,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": None, "recall": None, "f1": None, "roc_auc": None,
                "note": "只有单一类别"
            }

        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = None

        return {
            "n_samples": len(y_true),
            "n_yes": n_yes, "n_no": n_no, "n_unknown": n_unknown,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": auc,
        }

    def evaluate_all(self):
        """评估全部结果"""
        print("\n" + "=" * 60)
        print("全局评估")
        print("=" * 60)

        merge_metrics = {}
        review_metrics = {}

        for pt in PROMPT_TYPES:
            merge_metrics[pt] = {}
            review_metrics[pt] = {}

            for ct in CONTEXT_TYPES:
                merge_subset = [r for r in self.all_results
                                if r["task"] == "merge_prediction"
                                and r["prompt_type"] == pt
                                and r["context_type"] == ct]
                if merge_subset:
                    merge_metrics[pt][ct] = self._compute_merge_metrics(merge_subset)
                else:
                    merge_metrics[pt][ct] = None

                review_subset = [r for r in self.all_results
                                 if r["task"] == "review_comment"
                                 and r["prompt_type"] == pt
                                 and r["context_type"] == ct]
                if review_subset:
                    review_metrics[pt][ct] = self._compute_review_metrics(review_subset)
                else:
                    review_metrics[pt][ct] = None

        self.summary["merge_metrics"] = merge_metrics
        self.summary["review_metrics"] = review_metrics
        return merge_metrics, review_metrics

    def _compute_review_metrics(self, results):
        """计算 Review Comment 的 BLEU / ROUGE-L"""
        gt_map = {}
        for ctx in self.contexts:
            pr_id = ctx["pr_id"]
            comments = [
                rc.get("body", "").strip()
                for rc in ctx.get("review_comments", [])
                if rc.get("body", "").strip()
            ]
            if comments:
                gt_map[pr_id] = comments

        bleu_scores = []
        rouge_f1_scores = []
        n_with_gt = 0

        for r in results:
            pr_id = r["pr_id"]
            gt_comments = gt_map.get(pr_id, [])
            candidate = r.get("output_comment", "").strip()
            if not candidate or not gt_comments:
                continue

            n_with_gt += 1
            cand_tokens = _tokenize(candidate)
            best_bleu = 0.0
            best_rouge = 0.0
            for gt in gt_comments:
                ref_tokens = _tokenize(gt)
                bleu = compute_bleu(ref_tokens, cand_tokens)
                rouge = compute_rouge_l(ref_tokens, cand_tokens)
                best_bleu = max(best_bleu, bleu)
                best_rouge = max(best_rouge, rouge["f1"])
            bleu_scores.append(best_bleu)
            rouge_f1_scores.append(best_rouge)

        return {
            "n_samples": len(results),
            "n_with_gt": n_with_gt,
            "bleu_avg": np.mean(bleu_scores) if bleu_scores else 0.0,
            "rouge_l_avg": np.mean(rouge_f1_scores) if rouge_f1_scores else 0.0,
        }

    def print_merge_table(self):
        """打印 Merge Prediction 结果表格"""
        metrics = self.summary.get("merge_metrics", {})
        if not metrics:
            return
        print("\n" + "=" * 80)
        print("Merge Prediction 结果汇总")
        print("=" * 80)
        for pt in PROMPT_TYPES:
            print(f"\n--- {pt} ---")
            print(f"  {'Context':<16} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8} {'N':<6}")
            print(f"  {'-'*60}")
            for ct in CONTEXT_TYPES:
                m = metrics.get(pt, {}).get(ct)
                if m is None:
                    print(f"  {ct:<16} {'N/A':>8}")
                elif m.get("note"):
                    print(f"  {ct:<16} {m['accuracy']:<8.4f} ({m['note']})")
                else:
                    auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
                    prec_str = f"{m['precision']:.4f}" if m['precision'] is not None else "N/A"
                    rec_str = f"{m['recall']:.4f}" if m['recall'] is not None else "N/A"
                    f1_str = f"{m['f1']:.4f}" if m['f1'] is not None else "N/A"
                    print(f"  {ct:<16} {m['accuracy']:<8.4f} {prec_str:<8} "
                          f"{rec_str:<8} {f1_str:<8} {auc_str:<8} {m['n_samples']:<6}")

    def print_review_table(self):
        """打印 Review Comment 结果表格"""
        metrics = self.summary.get("review_metrics", {})
        if not metrics:
            return
        print("\n" + "=" * 80)
        print("Review Comment 结果汇总 (BLEU / ROUGE-L)")
        print("=" * 80)
        for pt in PROMPT_TYPES:
            print(f"\n--- {pt} ---")
            print(f"  {'Context':<16} {'BLEU':<10} {'ROUGE-L':<10} {'N':<6} {'N_GT':<6}")
            print(f"  {'-'*50}")
            for ct in CONTEXT_TYPES:
                m = metrics.get(pt, {}).get(ct)
                if m is None:
                    print(f"  {ct:<16} {'N/A':>10}")
                else:
                    print(f"  {ct:<16} {m['bleu_avg']:<10.4f} {m['rouge_l_avg']:<10.4f} "
                          f"{m['n_samples']:<6} {m['n_with_gt']:<6}")

    def save_summary(self):
        with open(STEP4_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
        print(f"\n汇总结果已保存: {STEP4_SUMMARY_PATH}")

    def run_all(self):
        """执行完整流程"""
        print("=" * 60)
        print("步骤四: LLM 代码审查测试 (Ollama)")
        print("=" * 60)

        self.load_contexts()
        self.run_all_experiments()
        self.evaluate_all()
        self.print_merge_table()
        self.print_review_table()
        self.save_summary()

        print("\n步骤四完成!")
        return self.summary

    def print_summary(self):
        print("\n" + "=" * 60)
        print("步骤四 LLM 结果摘要")
        print("=" * 60)
        self.print_merge_table()
        self.print_review_table()


if __name__ == "__main__":
    evaluator = LLMEvaluator()
    evaluator.run_all()