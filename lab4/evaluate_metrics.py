"""
阶段五评估工具：读取 results/ 目录下的 JSON 结果，统计各组指标。
可直接运行，无需依赖 notebook。

用法:
    python evaluate_metrics.py
"""
import json
import math
import os
import sys
from collections import Counter

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

CONTEXT_TYPES = ["diff_only", "diff_pr_desc", "diff_commit", "diff_extra"]
PROMPT_TYPES = ["zero_shot", "few_shot", "cot", "role_based"]


# ============================================================
#  数据加载
# ============================================================

def load_labels():
    """加载 selected_prs.json 中的 ground truth 标签"""
    with open(config.SELECTED_PRS_PATH, "r", encoding="utf-8") as f:
        prs = json.load(f)
    return {p["pr_id"]: p["label"] for p in prs}


def load_review_ground_truth():
    """加载 selected_prs.json 中的真实 review comments（按 pr_id 索引）"""
    with open(config.SELECTED_PRS_PATH, "r", encoding="utf-8") as f:
        prs = json.load(f)
    gt = {}
    for p in prs:
        comments = p.get("review_comments", [])
        if comments:
            gt[p["pr_id"]] = [c["body"] for c in comments if c.get("body", "").strip()]
    return gt


def load_results(task, prompt_type):
    """加载某一组的结果文件"""
    path = os.path.join(config.RESULTS_DIR, f"{task}_{prompt_type}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
#  Merge Prediction 评估
# ============================================================

def compute_merge_metrics(results, label_map):
    """计算 Merge Prediction 的评估指标"""
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
            pred = 0
            score = 0.5
            n_unknown += 1

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


# ============================================================
#  Review Comment 评估（BLEU + ROUGE）
# ============================================================

def _tokenize(text):
    """简单分词：按空白字符拆分并转小写"""
    return text.lower().split()


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(reference_tokens, candidate_tokens, max_n=4):
    """计算 BLEU 分数（带 brevity penalty）"""
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

    # Brevity penalty
    bp = 1.0 if len(candidate_tokens) >= len(reference_tokens) else math.exp(1 - len(reference_tokens) / max(len(candidate_tokens), 1))

    # Geometric mean of n-gram precisions
    log_avg = sum(math.log(p) for p in precisions if p > 0) / len(precisions)
    return bp * math.exp(log_avg)


def compute_rouge_l(reference_tokens, candidate_tokens):
    """计算 ROUGE-L（最长公共子序列）"""
    m, n = len(reference_tokens), len(candidate_tokens)
    if m == 0 or n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # LCS 长度
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


def compute_review_metrics(results, gt_map):
    """
    计算 Review Comment 的评估指标（BLEU 和 ROUGE-L）。

    对于每条 PR，将模型输出与真实 review comments 逐条对比，
    取最高分作为该 PR 的得分，然后对所有 PR 取平均。
    """
    bleu_scores = []
    rouge_f1_scores = []
    n_gt_empty = 0
    n_with_gt = 0

    for r in results:
        pr_id = r["pr_id"]
        gt_comments = gt_map.get(pr_id, [])
        candidate = r.get("output_comment", "").strip()

        if not candidate:
            continue

        if not gt_comments:
            n_gt_empty += 1
            continue

        n_with_gt += 1

        # 对每条 ground truth 计算得分，取最高
        best_bleu = 0.0
        best_rouge = 0.0
        candidate_tokens = _tokenize(candidate)

        for gt_text in gt_comments:
            ref_tokens = _tokenize(gt_text)
            if not ref_tokens:
                continue
            bleu = compute_bleu(ref_tokens, candidate_tokens)
            rouge = compute_rouge_l(ref_tokens, candidate_tokens)
            best_bleu = max(best_bleu, bleu)
            best_rouge = max(best_rouge, rouge["f1"])

        bleu_scores.append(best_bleu)
        rouge_f1_scores.append(best_rouge)

    if not bleu_scores:
        return None

    def mean_std(lst):
        avg = sum(lst) / len(lst)
        var = sum((x - avg) ** 2 for x in lst) / len(lst)
        return avg, math.sqrt(var)

    bleu_avg, bleu_std = mean_std(bleu_scores)
    rouge_avg, rouge_std = mean_std(rouge_f1_scores)

    return {
        "n_total": len(results),
        "n_with_gt": n_with_gt,
        "n_gt_empty": n_gt_empty,
        "bleu_mean": bleu_avg,
        "bleu_std": bleu_std,
        "rouge_l_mean": rouge_avg,
        "rouge_l_std": rouge_std,
    }


# ============================================================
#  主函数
# ============================================================

def print_merge_metrics():
    """打印 Merge Prediction 各组指标"""
    label_map = load_labels()

    print("=" * 70)
    print("Merge Prediction 各组评估指标")
    print("=" * 70)

    for pt in PROMPT_TYPES:
        results = load_results("merge_prediction", pt)
        if results is None:
            print(f"\n[{pt}] 文件不存在，跳过")
            continue

        m = compute_merge_metrics(results, label_map)
        if m is None:
            print(f"\n[{pt}] 无有效数据")
            continue

        print(f"\n{'─' * 50}")
        print(f"  Prompt: {pt}  ({m['n_total']} 条记录)")
        print(f"{'─' * 50}")
        print(f"  预测分布: Yes={m['n_yes']}, No={m['n_no']}, Unknown={m['n_unknown']}")
        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall   : {m['recall']:.4f}")
        print(f"  F1-score : {m['f1']:.4f}")
        if m["roc_auc"] is not None:
            print(f"  ROC-AUC  : {m['roc_auc']:.4f}")
        else:
            print(f"  ROC-AUC  : N/A")


def print_merge_cross_distribution():
    """打印上下文 × Prompt 的 Yes/No/Unknown 分布"""
    print(f"\n{'=' * 70}")
    print("合并预测分布统计（按上下文 × Prompt 交叉）")
    print(f"{'=' * 70}")

    for pt in PROMPT_TYPES:
        results = load_results("merge_prediction", pt)
        if results is None:
            continue

        print(f"\n--- {pt} ---")
        print(f"{'Context':<16} {'Yes':>6} {'No':>6} {'Unk':>6} {'Total':>6}")
        print("-" * 42)
        for ct in CONTEXT_TYPES:
            subset = [r for r in results if r["context_type"] == ct]
            yes = sum(1 for r in subset if r.get("output_decision") == "Yes")
            no = sum(1 for r in subset if r.get("output_decision") == "No")
            unk = len(subset) - yes - no
            print(f"{ct:<16} {yes:>6} {no:>6} {unk:>6} {len(subset):>6}")


def print_merge_accuracy_matrix():
    """打印上下文 × Prompt 的 Accuracy 矩阵"""
    label_map = load_labels()

    print(f"\n{'=' * 70}")
    print("按上下文 × Prompt 的 Accuracy 矩阵")
    print(f"{'=' * 70}")

    print(f"\n{'Context':<16}", end="")
    for pt in PROMPT_TYPES:
        print(f"{pt:<14}", end="")
    print()

    print("-" * (16 + 14 * len(PROMPT_TYPES)))
    for ct in CONTEXT_TYPES:
        print(f"{ct:<16}", end="")
        for pt in PROMPT_TYPES:
            results = load_results("merge_prediction", pt)
            if results is None:
                print(f"{'N/A':<14}", end="")
                continue
            subset = [r for r in results if r["context_type"] == ct]
            m = compute_merge_metrics(subset, label_map)
            if m:
                print(f"{m['accuracy']:<14.4f}", end="")
            else:
                print(f"{'N/A':<14}", end="")
        print()


def print_review_metrics():
    """打印 Review Comment 各组指标"""
    gt_map = load_review_ground_truth()

    print(f"\n{'=' * 70}")
    print("Review Comment 各组评估指标（BLEU + ROUGE-L）")
    print(f"{'=' * 70}")

    for pt in PROMPT_TYPES:
        results = load_results("review_comment", pt)
        if results is None:
            print(f"\n[{pt}] 文件不存在，跳过")
            continue

        m = compute_review_metrics(results, gt_map)
        if m is None:
            print(f"\n[{pt}] 无有效数据")
            continue

        print(f"\n{'─' * 50}")
        print(f"  Prompt: {pt}  ({m['n_total']} 条记录)")
        print(f"{'─' * 50}")
        print(f"  有真实评论的 PR 数: {m['n_with_gt']}")
        print(f"  无真实评论的 PR 数: {m['n_gt_empty']}")
        print(f"  BLEU-4  (mean ± std): {m['bleu_mean']:.4f} ± {m['bleu_std']:.4f}")
        print(f"  ROUGE-L (mean ± std): {m['rouge_l_mean']:.4f} ± {m['rouge_l_std']:.4f}")

    print(f"\n{'=' * 70}")
    print("按上下文 × Prompt 的 BLEU-4 矩阵")
    print(f"{'=' * 70}")

    print(f"\n{'Context':<16}", end="")
    for pt in PROMPT_TYPES:
        print(f"{pt:<14}", end="")
    print()
    print("-" * (16 + 14 * len(PROMPT_TYPES)))

    for ct in CONTEXT_TYPES:
        print(f"{ct:<16}", end="")
        for pt in PROMPT_TYPES:
            results = load_results("review_comment", pt)
            if results is None:
                print(f"{'N/A':<14}", end="")
                continue
            subset = [r for r in results if r["context_type"] == ct]
            m = compute_review_metrics(subset, gt_map)
            if m:
                print(f"{m['bleu_mean']:<14.4f}", end="")
            else:
                print(f"{'N/A':<14}", end="")
        print()

    print(f"\n{'=' * 70}")
    print("按上下文 × Prompt 的 ROUGE-L 矩阵")
    print(f"{'=' * 70}")

    print(f"\n{'Context':<16}", end="")
    for pt in PROMPT_TYPES:
        print(f"{pt:<14}", end="")
    print()
    print("-" * (16 + 14 * len(PROMPT_TYPES)))

    for ct in CONTEXT_TYPES:
        print(f"{ct:<16}", end="")
        for pt in PROMPT_TYPES:
            results = load_results("review_comment", pt)
            if results is None:
                print(f"{'N/A':<14}", end="")
                continue
            subset = [r for r in results if r["context_type"] == ct]
            m = compute_review_metrics(subset, gt_map)
            if m:
                print(f"{m['rouge_l_mean']:<14.4f}", end="")
            else:
                print(f"{'N/A':<14}", end="")
        print()


def print_review_examples():
    """打印模型生成的审查意见样例（与真实评论对比）"""
    gt_map = load_review_ground_truth()

    print(f"\n{'=' * 70}")
    print("模型生成审查意见 vs 真实评论 样例")
    print(f"{'=' * 70}")

    results = load_results("review_comment", "zero_shot")
    if results is None:
        print("zero_shot 结果文件不存在，跳过")
        return

    shown = 0
    for r in results:
        if shown >= 3:
            break
        pr_id = r["pr_id"]
        gt_comments = gt_map.get(pr_id, [])
        if not gt_comments:
            continue
        candidate = r.get("output_comment", "").strip()
        if not candidate:
            continue

        shown += 1
        print(f"\n{'─' * 50}")
        print(f"  PR #{pr_id} | {r['context_type']} | {r['prompt_type']}")
        print(f"{'─' * 50}")
        print(f"  模型生成:")
        print(f"  {candidate[:300]}...")
        print(f"  真实评论 ({len(gt_comments)} 条):")
        for i, gt in enumerate(gt_comments[:2]):
            print(f"    [{i+1}] {gt[:200]}...")
        print()


def main():
    # ---- Merge Prediction ----
    print_merge_metrics()
    print_merge_cross_distribution()
    print_merge_accuracy_matrix()

    # ---- Review Comment ----
    print_review_metrics()
    print_review_examples()


if __name__ == "__main__":
    main()