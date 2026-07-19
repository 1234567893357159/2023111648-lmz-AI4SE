"""
实验六 步骤五：结果分析模块
分析 LLM 评估结果，计算解析成功率、分类指标、BLEU/ROUGE-L，
并与实验五结果进行对比。

直接运行: python analyze_results.py
"""

import json
import math
import os
from collections import Counter

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import config

PROMPT_TYPES = config.PROMPT_TYPES
TASKS = config.TASK_TYPES
CONTEXT_TYPES = config.CONTEXT_TYPES

STEP4_RESULTS_DIR = os.path.join(config.LAB6_RESULTS_DIR, "step4")
LAB5_STEP4_RESULTS_DIR = os.path.join(config.LAB5_BASE_DIR, "results", "step4")


def _get_result_path(task, prompt_type):
    return os.path.join(STEP4_RESULTS_DIR, f"{task}_{prompt_type}.json")


def _get_lab5_result_path(task, prompt_type):
    return os.path.join(LAB5_STEP4_RESULTS_DIR, f"{task}_{prompt_type}.json")


# ========== BLEU / ROUGE-L ==========

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


# ========== 分类指标 ==========

def compute_classification_metrics(data):
    y_true, y_pred, y_score = [], [], []
    for r in data:
        label = r.get("label")
        if label is None:
            continue

        decision = r.get("output_decision", "Unknown")
        if decision == "Yes":
            y_true.append(label)
            y_pred.append(1)
            y_score.append(1.0)
        elif decision == "No":
            y_true.append(label)
            y_pred.append(0)
            y_score.append(0.0)
        else:
            continue

    n = len(y_true)
    if n == 0:
        return None

    if len(set(y_true)) < 2:
        return {
            "n_samples": n,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "note": "只有单一类别",
        }

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = None

    return {
        "n_samples": n,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def _format_cls(val):
    if val is None:
        return "N/A"
    return f"{val:.4f}"


# ========== 单组分析 ==========

def analyze_merge_prediction(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    parse_ok = sum(1 for r in data if r.get("parse_error") == False)
    parse_fail = sum(1 for r in data if r.get("parse_error") == True)
    unknown = sum(1 for r in data if r.get("output_decision") == "Unknown")
    yes = sum(1 for r in data if r.get("output_decision") == "Yes")
    no = sum(1 for r in data if r.get("output_decision") == "No")

    fail_records = [r for r in data if r.get("parse_error") == True]
    fail_yes = sum(1 for r in fail_records if r.get("output_decision") == "Yes")
    fail_no = sum(1 for r in fail_records if r.get("output_decision") == "No")
    fail_unknown = sum(1 for r in fail_records if r.get("output_decision") == "Unknown")

    ctx_stats = {}
    for ct in CONTEXT_TYPES:
        ctx_data = [r for r in data if r.get("context_type") == ct]
        ctx_total = len(ctx_data)
        ctx_ok = sum(1 for r in ctx_data if r.get("parse_error") == False)
        ctx_unk = sum(1 for r in ctx_data if r.get("output_decision") == "Unknown")
        ctx_cls = compute_classification_metrics(ctx_data)
        ctx_stats[ct] = {
            "total": ctx_total,
            "ok": ctx_ok,
            "fail": ctx_total - ctx_ok,
            "unknown": ctx_unk,
            "cls_metrics": ctx_cls,
        }

    avg_latency = sum(r.get("latency", 0) for r in data) / total if total > 0 else 0
    overall_cls = compute_classification_metrics(data)

    return {
        "total": total,
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
        "unknown": unknown,
        "yes": yes,
        "no": no,
        "fail_yes": fail_yes,
        "fail_no": fail_no,
        "fail_unknown": fail_unknown,
        "ctx_stats": ctx_stats,
        "avg_latency": avg_latency,
        "cls_metrics": overall_cls,
    }


def analyze_review_comment(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    avg_latency = sum(r.get("latency", 0) for r in data) / total if total > 0 else 0

    ctx_stats = {}
    for ct in CONTEXT_TYPES:
        ctx_data = [r for r in data if r.get("context_type") == ct]
        ctx_stats[ct] = {"total": len(ctx_data)}

    return {
        "total": total,
        "avg_latency": avg_latency,
        "ctx_stats": ctx_stats,
    }


# ========== 打印函数 ==========

def print_merge_section(stats, prompt_type):
    print(f"\n{'─' * 75}")
    print(f"  Prompt 类型: {prompt_type}")
    print(f"{'─' * 75}")

    print(f"\n  整体解析状况:")
    print(f"     总记录数:        {stats['total']}")
    print(f"     解析成功 (JSON):  {stats['parse_ok']:>4}  ({stats['parse_ok']/stats['total']*100:5.1f}%)")
    print(f"     解析失败:         {stats['parse_fail']:>4}  ({stats['parse_fail']/stats['total']*100:5.1f}%)")
    print(f"       └ 降级匹配到 Yes: {stats['fail_yes']:>4}")
    print(f"       └ 降级匹配到 No:  {stats['fail_no']:>4}")
    print(f"       └ 完全无法解析:   {stats['fail_unknown']:>4}")

    print(f"\n  预测分布:")
    print(f"     Yes:     {stats['yes']:>4}  ({stats['yes']/stats['total']*100:5.1f}%)")
    print(f"     No:      {stats['no']:>4}  ({stats['no']/stats['total']*100:5.1f}%)")
    print(f"     Unknown: {stats['unknown']:>4}  ({stats['unknown']/stats['total']*100:5.1f}%)")

    print(f"\n  按上下文类型拆分 (解析状况):")
    print(f"     {'上下文':<18} {'总数':<6} {'解析成功':<10} {'成功率':<10} {'Unknown':<8}")
    print(f"     {'─'*52}")
    for ct in CONTEXT_TYPES:
        cs = stats["ctx_stats"][ct]
        print(f"     {ct:<18} {cs['total']:<6} {cs['ok']:<10} {cs['ok']/cs['total']*100:<9.1f}% {cs['unknown']:<8}")

    print(f"\n  ⏱ 平均延迟: {stats['avg_latency']:.1f}s")

    cls = stats.get("cls_metrics")
    if cls:
        print(f"\n  整体分类指标 (vs 真实标签):")
        print(f"     样本数:    {cls['n_samples']}")
        print(f"     Accuracy:  {_format_cls(cls['accuracy'])}")
        print(f"     Precision: {_format_cls(cls['precision'])}")
        print(f"     Recall:    {_format_cls(cls['recall'])}")
        print(f"     F1-score:  {_format_cls(cls['f1'])}")
        print(f"     ROC-AUC:   {_format_cls(cls['roc_auc'])}")
        if cls.get("note"):
            print(f"     ⚠ {cls['note']}")

        print(f"\n  按上下文类型拆分 (分类指标):")
        print(f"     {'上下文':<18} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8} {'N':<6}")
        print(f"     {'─'*62}")
        for ct in CONTEXT_TYPES:
            cs = stats["ctx_stats"][ct]
            cm = cs.get("cls_metrics")
            if cm is None:
                print(f"     {ct:<18} {'N/A':>8}")
            elif cm.get("note"):
                print(f"     {ct:<18} {cm['accuracy']:<8.4f} ({cm['note']})")
            else:
                print(f"     {ct:<18} {_format_cls(cm['accuracy']):<8} "
                      f"{_format_cls(cm['precision']):<8} "
                      f"{_format_cls(cm['recall']):<8} "
                      f"{_format_cls(cm['f1']):<8} "
                      f"{_format_cls(cm['roc_auc']):<8} {cm['n_samples']:<6}")


def print_merge_summary_table(all_merge_stats):
    print(f"\n\n{'=' * 75}")
    print(f"  Merge Prediction 跨 Prompt 汇总对比")
    print(f"{'=' * 75}")

    print(f"\n  {'Prompt':<16} {'总数':<6} {'解析成功':<10} {'成功率':<10} {'Unknown':<10} {'Yes':<8} {'No':<8} {'延迟':<8}")
    print(f"  {'─'*75}")
    for pt in PROMPT_TYPES:
        if pt not in all_merge_stats:
            continue
        s = all_merge_stats[pt]
        print(f"  {pt:<16} {s['total']:<6} {s['parse_ok']:<10} "
              f"{s['parse_ok']/s['total']*100:<9.1f}% {s['unknown']:<10} "
              f"{s['yes']:<8} {s['no']:<8} {s['avg_latency']:<7.1f}s")

    print(f"\n\n{'=' * 75}")
    print(f"  Merge Prediction 跨 Prompt × 上下文 分类指标对比")
    print(f"{'=' * 75}")
    for metric_name, metric_key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                                     ("Recall", "recall"), ("F1-score", "f1"),
                                     ("ROC-AUC", "roc_auc")]:
        print(f"\n  {metric_name}:")
        print(f"  {'Prompt':<16}", end="")
        for ct in CONTEXT_TYPES:
            print(f"{ct:<16}", end="")
        print()
        print(f"  {'─'*80}")
        for pt in PROMPT_TYPES:
            if pt not in all_merge_stats:
                continue
            print(f"  {pt:<16}", end="")
            for ct in CONTEXT_TYPES:
                cs = all_merge_stats[pt]["ctx_stats"][ct]
                cm = cs.get("cls_metrics")
                if cm is None or cm.get(metric_key) is None:
                    print(f"{'N/A':<16}", end="")
                else:
                    print(f"{cm[metric_key]:<16.4f}", end="")
            print()


def print_unknown_source_analysis(all_merge_stats):
    print(f"\n\n{'=' * 75}")
    print(f"  Merge Prediction Unknown 来源分析")
    print(f"{'=' * 75}")

    print(f"\n  {'Prompt':<16}", end="")
    for ct in CONTEXT_TYPES:
        print(f"{ct:<16}", end="")
    print(f"{'合计':<8}")
    print(f"  {'─'*80}")
    for pt in PROMPT_TYPES:
        if pt not in all_merge_stats:
            continue
        print(f"  {pt:<16}", end="")
        total_unk = 0
        for ct in CONTEXT_TYPES:
            unk = all_merge_stats[pt]["ctx_stats"][ct]["unknown"]
            print(f"{unk:<16}", end="")
            total_unk += unk
        print(f"{total_unk:<8}")


def print_review_section(stats, prompt_type):
    print(f"\n  {prompt_type:<16} 总记录: {stats['total']}, 平均延迟: {stats['avg_latency']:.1f}s")


def print_review_summary_table(all_review_stats):
    print(f"\n\n{'=' * 75}")
    print(f"  Review Comment 跨 Prompt 汇总对比")
    print(f"{'=' * 75}")

    print(f"\n  {'Prompt':<16} {'总数':<6} {'延迟':<10}", end="")
    for ct in CONTEXT_TYPES:
        print(f"{ct:<16}", end="")
    print()
    print(f"  {'─'*80}")
    for pt in PROMPT_TYPES:
        if pt not in all_review_stats:
            continue
        s = all_review_stats[pt]
        print(f"  {pt:<16} {s['total']:<6} {s['avg_latency']:<9.1f}s", end="")
        for ct in CONTEXT_TYPES:
            cs = s["ctx_stats"].get(ct, {})
            print(f"{cs.get('total', 0):<16}", end="")
        print()


# ========== 与 lab5 对比 ==========

def load_lab5_results():
    """加载 lab5 step4 的全部结果"""
    lab5_merge = {}
    lab5_review = {}
    lab5_prompt_types = ["zero_shot", "few_shot", "cot", "role_based"]
    lab5_context_types = ["diff_only", "diff_pr_desc", "diff_commit", "diff_extra"]

    for pt in lab5_prompt_types:
        path = _get_lab5_result_path("merge_prediction", pt)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lab5_merge[pt] = json.load(f)

    for pt in lab5_prompt_types:
        path = _get_lab5_result_path("review_comment", pt)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lab5_review[pt] = json.load(f)

    return lab5_merge, lab5_review


def compare_merge_with_lab5(all_merge_stats, lab5_merge_data):
    """对比 lab6 和 lab5 的 Merge Prediction 结果"""
    lab5_merge_stats = {}
    lab5_prompt_types = ["zero_shot", "few_shot", "cot", "role_based"]
    lab5_context_types = ["diff_only", "diff_pr_desc", "diff_commit", "diff_extra"]

    for pt in lab5_prompt_types:
        if pt not in lab5_merge_data:
            continue
        data = lab5_merge_data[pt]
        total = len(data)
        parse_ok = sum(1 for r in data if r.get("parse_error") == False)
        parse_fail = sum(1 for r in data if r.get("parse_error") == True)
        unknown = sum(1 for r in data if r.get("output_decision") == "Unknown")
        avg_lat = sum(r.get("latency", 0) for r in data) / total if total > 0 else 0

        ctx_stats = {}
        for ct in lab5_context_types:
            ctx_data = [r for r in data if r.get("context_type") == ct]
            ctx_cls = compute_classification_metrics(ctx_data)
            ctx_stats[ct] = {
                "total": len(ctx_data),
                "cls_metrics": ctx_cls,
            }

        lab5_merge_stats[pt] = {
            "total": total,
            "parse_ok": parse_ok,
            "parse_fail": parse_fail,
            "unknown": unknown,
            "avg_latency": avg_lat,
            "ctx_stats": ctx_stats,
            "cls_metrics": compute_classification_metrics(data),
        }

    print(f"\n\n{'=' * 75}")
    print(f"  lab6 vs lab5 Merge Prediction 对比")
    print(f"{'=' * 75}")

    prompt_mapping = {
        "few_shot": "few_shot",
        "cot": "cot",
        "role_based": "role_based",
        "self_reflection": "zero_shot",
    }
    context_mapping = {
        "diff_only": "diff_only",
        "diff_pr_desc": "diff_pr_desc",
        "diff_repo": "diff_extra",
        "diff_issue": "diff_extra",
        "diff_full": "diff_extra",
    }

    print(f"\n  说明: lab6 的 self_reflection 与 lab5 的 zero_shot 对比")
    print(f"        lab6 的 diff_repo/diff_issue/diff_full 与 lab5 的 diff_extra 对比")
    print(f"        lab6 的 diff_commit 和 diff_extra 已移除\n")

    for metric_name, metric_key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                                     ("Recall", "recall"), ("F1-score", "f1"),
                                     ("ROC-AUC", "roc_auc")]:
        print(f"\n  {metric_name} (lab6 vs lab5):")
        print(f"  {'Prompt':<16} {'上下文':<18} {'lab6':<10} {'lab5':<10} {'Delta':<10}")
        print(f"  {'─'*64}")

        for pt in PROMPT_TYPES:
            l5_pt = prompt_mapping.get(pt)
            if pt not in all_merge_stats or l5_pt not in lab5_merge_stats:
                continue

            for ct in CONTEXT_TYPES:
                l5_ct = context_mapping.get(ct)
                if l5_ct is None:
                    continue

                l6_cm = all_merge_stats[pt]["ctx_stats"][ct].get("cls_metrics")
                l5_cm = lab5_merge_stats[l5_pt]["ctx_stats"].get(l5_ct, {}).get("cls_metrics")

                if l6_cm is None or l5_cm is None:
                    continue

                v6 = l6_cm.get(metric_key)
                v5 = l5_cm.get(metric_key)
                if v6 is None or v5 is None:
                    continue

                delta = v6 - v5
                sign = "+" if delta >= 0 else ""
                print(f"  {pt:<16} {ct:<18} {v6:<10.4f} {v5:<10.4f} {sign}{delta:<9.4f}")


def compare_review_with_lab5(all_review_stats, lab5_review_data):
    """对比 lab6 和 lab5 的 Review Comment 结果"""
    print(f"\n\n{'=' * 75}")
    print(f"  lab6 vs lab5 Review Comment 延迟对比")
    print(f"{'=' * 75}")

    lab5_review_stats = {}
    lab5_prompt_types = ["zero_shot", "few_shot", "cot", "role_based"]

    for pt in lab5_prompt_types:
        if pt in lab5_review_data:
            data = lab5_review_data[pt]
            total = len(data)
            avg_lat = sum(r.get("latency", 0) for r in data) / total if total > 0 else 0
            lab5_review_stats[pt] = {"total": total, "avg_latency": avg_lat}

    prompt_mapping = {
        "few_shot": "few_shot",
        "cot": "cot",
        "role_based": "role_based",
        "self_reflection": "zero_shot",
    }

    print(f"\n  {'Prompt':<16} {'lab6 延迟':<12} {'lab5 延迟':<12} {'Delta':<10}")
    print(f"  {'─'*50}")
    for pt in PROMPT_TYPES:
        l5_pt = prompt_mapping.get(pt)
        if pt not in all_review_stats or l5_pt not in lab5_review_stats:
            continue
        l6_lat = all_review_stats[pt]["avg_latency"]
        l5_lat = lab5_review_stats[l5_pt]["avg_latency"]
        delta = l6_lat - l5_lat
        sign = "+" if delta >= 0 else ""
        print(f"  {pt:<16} {l6_lat:<11.1f}s {l5_lat:<11.1f}s {sign}{delta:<9.1f}s")


# ========== 主入口 ==========

def run_analysis():
    """运行完整分析，返回 (all_merge_stats, all_review_stats)"""
    print("=" * 75)
    print("  实验六 步骤五：LLM 评估结果分析")
    print("=" * 75)

    all_merge_stats = {}
    all_review_stats = {}

    # ---- Merge Prediction ----
    print(f"\n\n{'#' * 75}")
    print(f"  Task: Merge Prediction")
    print(f"{'#' * 75}")

    for pt in PROMPT_TYPES:
        path = _get_result_path("merge_prediction", pt)
        if not os.path.exists(path):
            print(f"\n  ⚠ 文件不存在: {path}")
            continue
        stats = analyze_merge_prediction(path)
        all_merge_stats[pt] = stats
        print_merge_section(stats, pt)

    # ---- Review Comment ----
    print(f"\n\n{'#' * 75}")
    print(f"  Task: Review Comment")
    print(f"{'#' * 75}")

    for pt in PROMPT_TYPES:
        path = _get_result_path("review_comment", pt)
        if not os.path.exists(path):
            print(f"\n  ⚠ 文件不存在: {path}")
            continue
        stats = analyze_review_comment(path)
        all_review_stats[pt] = stats
        print_review_section(stats, pt)

    # ---- 跨 Prompt 汇总 ----
    print_merge_summary_table(all_merge_stats)
    print_unknown_source_analysis(all_merge_stats)
    print_review_summary_table(all_review_stats)

    # ---- 与 lab5 对比 ----
    lab5_merge, lab5_review = load_lab5_results()
    if lab5_merge:
        compare_merge_with_lab5(all_merge_stats, lab5_merge)
    if lab5_review:
        compare_review_with_lab5(all_review_stats, lab5_review)

    print(f"\n\n✅ 分析完成!")
    return all_merge_stats, all_review_stats


if __name__ == "__main__":
    run_analysis()