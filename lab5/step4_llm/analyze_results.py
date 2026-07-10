"""
分析 lab5 step4 LLM 评估结果的解析成功率 + 分类指标
支持 merge_prediction 和 review_comment 两个任务
直接运行即可: python analyze_results.py
"""
import json
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STEP4_RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results", "step4")

PROMPT_TYPES = ["zero_shot", "few_shot", "cot", "role_based"]
TASKS = ["merge_prediction", "review_comment"]
CONTEXT_TYPES = ["diff_only", "diff_pr_desc", "diff_commit", "diff_extra"]


def _get_result_path(task, prompt_type):
    return os.path.join(STEP4_RESULTS_DIR, f"{task}_{prompt_type}.json")


def compute_classification_metrics(data):
    """根据记录中的 label 字段计算分类指标"""
    y_true, y_pred, y_score = [], [], []
    for r in data:
        label = r.get("label")
        if label is None:
            continue
        y_true.append(label)

        decision = r.get("output_decision", "Unknown")
        if decision == "Yes":
            y_pred.append(1)
            y_score.append(1.0)
        elif decision == "No":
            y_pred.append(0)
            y_score.append(0.0)
        else:
            y_pred.append(0)
            y_score.append(0.5)

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


def analyze_merge_prediction(filepath):
    """分析 merge_prediction 结果文件"""
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
    """分析 review_comment 结果文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    avg_latency = sum(r.get("latency", 0) for r in data) / total if total > 0 else 0

    ctx_stats = {}
    for ct in CONTEXT_TYPES:
        ctx_data = [r for r in data if r.get("context_type") == ct]
        ctx_stats[ct] = {
            "total": len(ctx_data),
        }

    return {
        "total": total,
        "avg_latency": avg_latency,
        "ctx_stats": ctx_stats,
    }


def print_merge_section(stats, prompt_type):
    """打印单个 merge_prediction 的详细报告"""
    print(f"\n{'─' * 75}")
    print(f"  Prompt 类型: {prompt_type}")
    print(f"{'─' * 75}")

    print(f"\n  📊 整体解析状况:")
    print(f"     总记录数:        {stats['total']}")
    print(f"     解析成功 (JSON):  {stats['parse_ok']:>4}  ({stats['parse_ok']/stats['total']*100:5.1f}%)")
    print(f"     解析失败:         {stats['parse_fail']:>4}  ({stats['parse_fail']/stats['total']*100:5.1f}%)")
    print(f"       └ 降级匹配到 Yes: {stats['fail_yes']:>4}")
    print(f"       └ 降级匹配到 No:  {stats['fail_no']:>4}")
    print(f"       └ 完全无法解析:   {stats['fail_unknown']:>4}")

    print(f"\n  📈 预测分布:")
    print(f"     Yes:     {stats['yes']:>4}  ({stats['yes']/stats['total']*100:5.1f}%)")
    print(f"     No:      {stats['no']:>4}  ({stats['no']/stats['total']*100:5.1f}%)")
    print(f"     Unknown: {stats['unknown']:>4}  ({stats['unknown']/stats['total']*100:5.1f}%)")

    print(f"\n  🔍 按上下文类型拆分 (解析状况):")
    print(f"     {'上下文':<18} {'总数':<6} {'解析成功':<10} {'成功率':<10} {'Unknown':<8}")
    print(f"     {'─'*52}")
    for ct in CONTEXT_TYPES:
        cs = stats["ctx_stats"][ct]
        print(f"     {ct:<18} {cs['total']:<6} {cs['ok']:<10} {cs['ok']/cs['total']*100:<9.1f}% {cs['unknown']:<8}")

    print(f"\n  ⏱ 平均延迟: {stats['avg_latency']:.1f}s")

    cls = stats.get("cls_metrics")
    if cls:
        print(f"\n  🎯 整体分类指标 (vs 真实标签):")
        print(f"     样本数:    {cls['n_samples']}")
        print(f"     Accuracy:  {_format_cls(cls['accuracy'])}")
        print(f"     Precision: {_format_cls(cls['precision'])}")
        print(f"     Recall:    {_format_cls(cls['recall'])}")
        print(f"     F1-score:  {_format_cls(cls['f1'])}")
        print(f"     ROC-AUC:   {_format_cls(cls['roc_auc'])}")
        if cls.get("note"):
            print(f"     ⚠ {cls['note']}")

        print(f"\n  🔍 按上下文类型拆分 (分类指标):")
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
    """打印 merge_prediction 跨 Prompt 汇总对比表"""
    print(f"\n\n{'=' * 75}")
    print(f"  Merge Prediction 跨 Prompt 汇总对比")
    print(f"{'=' * 75}")

    print(f"\n  {'Prompt':<15} {'总数':<6} {'解析成功':<10} {'成功率':<10} {'Unknown':<10} {'Yes':<8} {'No':<8} {'延迟':<8}")
    print(f"  {'─'*75}")
    for pt in PROMPT_TYPES:
        if pt not in all_merge_stats:
            continue
        s = all_merge_stats[pt]
        print(f"  {pt:<15} {s['total']:<6} {s['parse_ok']:<10} "
              f"{s['parse_ok']/s['total']*100:<9.1f}% {s['unknown']:<10} "
              f"{s['yes']:<8} {s['no']:<8} {s['avg_latency']:<7.1f}s")

    print(f"\n\n{'=' * 75}")
    print(f"  Merge Prediction 跨 Prompt × 上下文 分类指标对比")
    print(f"{'=' * 75}")
    for metric_name, metric_key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                                     ("Recall", "recall"), ("F1-score", "f1"),
                                     ("ROC-AUC", "roc_auc")]:
        print(f"\n  📊 {metric_name}:")
        print(f"  {'Prompt':<15}", end="")
        for ct in CONTEXT_TYPES:
            print(f"{ct:<16}", end="")
        print()
        print(f"  {'─'*80}")
        for pt in PROMPT_TYPES:
            if pt not in all_merge_stats:
                continue
            print(f"  {pt:<15}", end="")
            for ct in CONTEXT_TYPES:
                cs = all_merge_stats[pt]["ctx_stats"][ct]
                cm = cs.get("cls_metrics")
                if cm is None or cm.get(metric_key) is None:
                    print(f"{'N/A':<16}", end="")
                else:
                    print(f"{cm[metric_key]:<16.4f}", end="")
            print()

    print(f"\n\n{'=' * 75}")
    print(f"  Merge Prediction Unknown 来源分析")
    print(f"{'=' * 75}")
    print(f"\n  {'Prompt':<15}", end="")
    for ct in CONTEXT_TYPES:
        print(f"{ct:<16}", end="")
    print(f"  {'合计':<6}")
    print(f"  {'─'*80}")
    for pt in PROMPT_TYPES:
        if pt not in all_merge_stats:
            continue
        print(f"  {pt:<15}", end="")
        total_unk = 0
        for ct in CONTEXT_TYPES:
            cs = all_merge_stats[pt]["ctx_stats"][ct]
            print(f"{cs['unknown']:<16}", end="")
            total_unk += cs["unknown"]
        print(f"  {total_unk:<6}")


def print_review_summary_table(all_review_stats):
    """打印 review_comment 跨 Prompt 汇总对比表"""
    print(f"\n\n{'=' * 75}")
    print(f"  Review Comment 跨 Prompt 汇总对比")
    print(f"{'=' * 75}")

    print(f"\n  {'Prompt':<15} {'总数':<6} {'延迟':<8}", end="")
    for ct in CONTEXT_TYPES:
        print(f"{ct:<16}", end="")
    print()
    print(f"  {'─'*80}")
    for pt in PROMPT_TYPES:
        if pt not in all_review_stats:
            continue
        s = all_review_stats[pt]
        print(f"  {pt:<15} {s['total']:<6} {s['avg_latency']:<7.1f}s", end="")
        for ct in CONTEXT_TYPES:
            cs = s["ctx_stats"][ct]
            print(f"{cs['total']:<16}", end="")
        print()


def main():
    print("=" * 75)
    print("  Lab5 Step4 LLM 评估结果分析")
    print("=" * 75)

    all_merge_stats = {}
    all_review_stats = {}

    for task in TASKS:
        task_label = "Merge Prediction" if task == "merge_prediction" else "Review Comment"
        print(f"\n\n{'#' * 75}")
        print(f"  Task: {task_label}")
        print(f"{'#' * 75}")

        for pt in PROMPT_TYPES:
            filepath = _get_result_path(task, pt)
            if not os.path.exists(filepath):
                print(f"\n  [跳过] 文件不存在: {os.path.basename(filepath)}")
                continue

            if task == "merge_prediction":
                stats = analyze_merge_prediction(filepath)
                all_merge_stats[pt] = stats
                print_merge_section(stats, pt)
            else:
                stats = analyze_review_comment(filepath)
                all_review_stats[pt] = stats
                print(f"\n  {pt:<15} 总记录: {stats['total']}, 平均延迟: {stats['avg_latency']:.1f}s")

    if all_merge_stats:
        print_merge_summary_table(all_merge_stats)

    if all_review_stats:
        print_review_summary_table(all_review_stats)

    print(f"\n\n✅ 分析完成!")


if __name__ == "__main__":
    main()